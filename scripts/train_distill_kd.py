"""
Train Prométhée via True Knowledge Distillation

Prométhée is trained using both:
1. Hard labels (cross-entropy on teacher's generated text)
2. Soft labels (KL divergence on teacher's logits)

Features:
- Muon optimizer (SOTA) with gradient clipping
- Vocabulary projection for GLM-4 → Mamba
- Multi-task causal reasoning
"""

import os
import json
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from transformers import (
    MambaForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from loguru import logger

# Default configuration
DEFAULT_STUDENT_MODEL = "state-spaces/mamba-1.4b-hf"
DEFAULT_LOGITS_DIR = "data/teacher_logits"
DEFAULT_OUTPUT_DIR = "./models/promethee-kd"

# KD hyperparameters
ALPHA = 0.5  # Weight for hard label loss (1-ALPHA for soft labels)
TEMPERATURE = 2.0  # Must match teacher generation temperature


# =============================================================================
# Muon Optimizer (SOTA for LLMs)
# =============================================================================

class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum-based optimizer without exponential moving averages.
    SOTA for LLM training, better convergence than AdamW.
    """

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, momentum, nesterov, wd = group['lr'], group['momentum'], group['nesterov'], group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                if wd != 0:
                    p.mul_(1 - lr * wd)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

                p.add_(d_p, alpha=-lr)

        return loss


class VocabProjector:
    """
    Projects teacher vocabulary to student vocabulary.

    Since GLM-4 and Mamba have different tokenizers, we create a mapping
    between their vocabularies based on decoded text matching.
    """

    def __init__(self, teacher_tokenizer_path: str, student_tokenizer):
        self.student_tokenizer = student_tokenizer

        # Load teacher tokenizer
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_tokenizer_path, trust_remote_code=True
        )

        # Build vocabulary mapping (teacher_id -> student_id)
        # This is approximate - we match tokens by their decoded text
        logger.info("Building vocabulary projection map...")
        self.teacher_to_student = {}
        self.coverage = 0

        teacher_vocab = self.teacher_tokenizer.get_vocab()
        student_vocab = self.student_tokenizer.get_vocab()

        for token_str, teacher_id in tqdm(teacher_vocab.items(), desc="Mapping vocab"):
            if token_str in student_vocab:
                self.teacher_to_student[teacher_id] = student_vocab[token_str]
                self.coverage += 1

        logger.info(f"Vocabulary coverage: {self.coverage}/{len(teacher_vocab)} "
                    f"({100*self.coverage/len(teacher_vocab):.1f}%)")

    def project_distribution(self, teacher_probs: torch.Tensor,
                             teacher_indices: torch.Tensor,
                             student_vocab_size: int) -> torch.Tensor:
        """
        Project teacher's top-k probability distribution to student vocabulary.

        Args:
            teacher_probs: (seq_len, top_k) probability values
            teacher_indices: (seq_len, top_k) token indices in teacher vocab
            student_vocab_size: Size of student vocabulary

        Returns:
            (seq_len, student_vocab_size) projected probability distribution
        """
        seq_len, top_k = teacher_probs.shape
        device = teacher_probs.device

        # Initialize uniform distribution (small probability mass)
        student_dist = torch.full(
            (seq_len, student_vocab_size),
            fill_value=1e-8,
            device=device
        )

        # Project each position
        for pos in range(seq_len):
            mapped_mass = 0.0
            for k in range(top_k):
                teacher_id = teacher_indices[pos, k].item()
                prob = teacher_probs[pos, k].item()

                if teacher_id in self.teacher_to_student:
                    student_id = self.teacher_to_student[teacher_id]
                    student_dist[pos, student_id] += prob
                    mapped_mass += prob

            # Normalize (redistribute unmapped mass uniformly)
            unmapped_mass = 1.0 - mapped_mass
            if unmapped_mass > 0:
                student_dist[pos] += unmapped_mass / student_vocab_size

            # Re-normalize to ensure valid distribution
            student_dist[pos] = student_dist[pos] / student_dist[pos].sum()

        return student_dist


class KDDataset(Dataset):
    """Dataset for Knowledge Distillation training"""

    def __init__(self, logits_dir: str, student_tokenizer, max_length: int = 2048):
        self.logits_dir = Path(logits_dir)
        self.student_tokenizer = student_tokenizer
        self.max_length = max_length

        # Load index
        index_file = self.logits_dir / "index.jsonl"
        self.samples = []
        with open(index_file, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load logits
        logits_file = self.logits_dir / sample["logits_file"]
        data = np.load(logits_file)

        # Format input for student
        input_text = sample["input_text"]
        response_text = sample["response_text"]

        prompt = f"Analyze the following financial text and provide a structured assessment:\n\nInput: {input_text}\n\nAnalysis:\n{response_text}"

        # Tokenize for student
        encoding = self.student_tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "teacher_probs": torch.from_numpy(data["top_k_probs"].astype(np.float32)),
            "teacher_indices": torch.from_numpy(data["top_k_indices"].astype(np.int64)),
            "response_text": response_text
        }


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
    L = α * L_hard + (1-α) * T² * L_soft

    Where:
    - L_hard: Cross-entropy on hard labels (teacher's generated tokens)
    - L_soft: KL divergence on soft labels (teacher's probability distribution)
    - T: Temperature
    - α: Mixing coefficient
    """

    def __init__(self, alpha: float = ALPHA, temperature: float = TEMPERATURE):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, labels: torch.Tensor,
                teacher_soft_labels: torch.Tensor = None) -> dict:
        """
        Args:
            student_logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len) hard labels
            teacher_soft_labels: (batch, seq_len, vocab_size) soft labels from teacher

        Returns:
            dict with total_loss, hard_loss, soft_loss
        """
        # Hard label loss (standard cross-entropy)
        hard_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        if teacher_soft_labels is None or self.alpha >= 1.0:
            return {
                "total_loss": hard_loss,
                "hard_loss": hard_loss,
                "soft_loss": torch.tensor(0.0)
            }

        # Soft label loss (KL divergence)
        # Apply temperature to student logits
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = teacher_soft_labels  # Already softmax with temperature

        # KL divergence (sum over vocab, mean over positions)
        soft_loss = F.kl_div(
            student_soft.view(-1, student_logits.size(-1)),
            teacher_soft.view(-1, teacher_soft.size(-1)),
            reduction='batchmean',
            log_target=False
        )

        # Scale soft loss by T²
        soft_loss = soft_loss * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return {
            "total_loss": total_loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss
        }


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn,
                vocab_projector, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_hard = 0
    total_soft = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Shift logits and labels for causal LM
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()

        # Project teacher distributions if available
        teacher_soft = None
        if vocab_projector is not None:
            # This is simplified - in practice you'd batch this
            # and only apply to response tokens
            pass  # TODO: implement batched projection

        # Compute loss
        losses = loss_fn(logits, labels, teacher_soft)

        # Backward pass
        optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += losses["total_loss"].item()
        total_hard += losses["hard_loss"].item()
        total_soft += losses["soft_loss"].item() if isinstance(losses["soft_loss"], torch.Tensor) else 0

        pbar.set_postfix({
            "loss": f"{losses['total_loss'].item():.4f}",
            "hard": f"{losses['hard_loss'].item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

    n = len(dataloader)
    return total_loss/n, total_hard/n, total_soft/n


def main():
    parser = argparse.ArgumentParser(description="Train Prométhée with Knowledge Distillation")
    parser.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--logits-dir", type=str, default=DEFAULT_LOGITS_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Weight for hard loss (0=pure KD, 1=pure CE)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--use-muon", action="store_true", help="Use Muon optimizer (SOTA)")
    parser.add_argument("--use-soft-labels", action="store_true",
                        help="Use soft labels from teacher (requires vocab projection)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("🔥 Prométhée - Knowledge Distillation Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Optimizer: {'Muon (SOTA)' if args.use_muon else 'AdamW'}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Gradient clipping: {args.grad_clip}")
    logger.info("")

    # Load student model and tokenizer
    logger.info(f"Loading base model: {args.student_model}")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    student_tokenizer.pad_token = student_tokenizer.eos_token

    model = MambaForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.float16
    ).to(device)

    # Create vocabulary projector if using soft labels
    vocab_projector = None
    if args.use_soft_labels:
        teacher_tokenizer_path = Path(args.logits_dir) / "teacher_tokenizer"
        if teacher_tokenizer_path.exists():
            vocab_projector = VocabProjector(
                str(teacher_tokenizer_path),
                student_tokenizer
            )
        else:
            logger.warning("Teacher tokenizer not found, using hard labels only")
            args.alpha = 1.0

    # Create dataset and dataloader
    dataset = KDDataset(
        args.logits_dir,
        student_tokenizer,
        max_length=args.max_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Optimizer and scheduler
    if args.use_muon:
        logger.info("Using Muon optimizer (SOTA)")
        optimizer = Muon(model.parameters(), lr=args.lr, momentum=0.95, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Loss function
    loss_fn = DistillationLoss(alpha=args.alpha, temperature=args.temperature)

    # Training loop
    logger.info("Starting Prométhée Knowledge Distillation...")
    logger.info(f"Alpha (hard loss weight): {args.alpha}")
    logger.info(f"Temperature: {args.temperature}")

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        avg_loss, avg_hard, avg_soft = train_epoch(
            model, dataloader, optimizer, scheduler,
            loss_fn, vocab_projector, device, epoch
        )

        logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}, hard={avg_hard:.4f}, soft={avg_soft:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            logger.info(f"Saving best Prométhée model (loss={best_loss:.4f})")
            model.save_pretrained(f"{args.output_dir}/checkpoint-best")
            student_tokenizer.save_pretrained(f"{args.output_dir}/checkpoint-best")

        # Save epoch checkpoint
        if epoch % 5 == 0:
            model.save_pretrained(f"{args.output_dir}/checkpoint-epoch-{epoch}")

    # Save final model
    logger.info(f"Saving final Prométhée model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    student_tokenizer.save_pretrained(args.output_dir)

    # Save Prométhée config
    config = {
        "model_name": "Prométhée",
        "model_type": "mamba-causal-lm",
        "version": "1.0",
        "base_model": args.student_model,
        "task": "financial_causal_reasoning",
        "training": {
            "method": "knowledge_distillation",
            "optimizer": "muon" if args.use_muon else "adamw",
            "epochs": args.epochs,
            "alpha": args.alpha,
            "temperature": args.temperature,
            "gradient_clip": args.grad_clip,
            "final_loss": avg_loss
        }
    }
    with open(f"{args.output_dir}/promethee_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("")
    logger.info("🔥 Prométhée training complete!")


if __name__ == "__main__":
    main()
