"""
Train Mamba Student Model via True Knowledge Distillation

This script trains Mamba using both:
1. Hard labels (cross-entropy on teacher's generated text)
2. Soft labels (KL divergence on teacher's logits)

Handles vocabulary mismatch between teacher (GLM-4) and student (Mamba)
via vocabulary projection.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from transformers import (
    MambaForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from loguru import logger

# Default configuration
DEFAULT_STUDENT_MODEL = "state-spaces/mamba-1.4b-hf"
DEFAULT_LOGITS_DIR = "data/teacher_logits"
DEFAULT_OUTPUT_DIR = "./models/fin-mamba-student-kd"

# KD hyperparameters
ALPHA = 0.5  # Weight for hard label loss (1-ALPHA for soft labels)
TEMPERATURE = 2.0  # Must match teacher generation temperature


class VocabProjector:
    """
    Projects teacher vocabulary to student vocabulary.

    Since GLM-4 and Mamba have different tokenizers, we create a mapping
    between their vocabularies based on decoded text matching.
    Precomputes a lookup tensor for fast batched projection.
    """

    def __init__(self, teacher_tokenizer_path: str, student_tokenizer):
        self.student_tokenizer = student_tokenizer

        # Load teacher tokenizer
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_tokenizer_path, trust_remote_code=True
        )

        # Build vocabulary mapping (teacher_id -> student_id)
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

        # Precompute lookup tensor: teacher_id -> student_id (-1 if unmapped)
        max_teacher_id = max(teacher_vocab.values()) + 1
        self._lookup = torch.full((max_teacher_id,), -1, dtype=torch.long)
        for t_id, s_id in self.teacher_to_student.items():
            self._lookup[t_id] = s_id

    def project_distribution_batched(self, teacher_probs: torch.Tensor,
                                     teacher_indices: torch.Tensor,
                                     student_vocab_size: int) -> torch.Tensor:
        """
        Vectorized projection of teacher top-k distributions to student vocabulary.

        Args:
            teacher_probs: (batch, seq_len, top_k) probability values
            teacher_indices: (batch, seq_len, top_k) token indices in teacher vocab
            student_vocab_size: Size of student vocabulary

        Returns:
            (batch, seq_len, student_vocab_size) projected probability distribution
        """
        B, S, K = teacher_probs.shape
        device = teacher_probs.device

        # Map teacher indices to student indices
        lookup = self._lookup.to(device)
        # Clamp to avoid out-of-bounds (unmapped tokens beyond lookup size)
        clamped_indices = teacher_indices.clamp(0, lookup.shape[0] - 1)
        student_indices = lookup[clamped_indices]  # (B, S, K), -1 for unmapped

        # Build output distribution
        student_dist = torch.full(
            (B, S, student_vocab_size), fill_value=1e-8,
            device=device, dtype=teacher_probs.dtype
        )

        # Mask for valid mappings
        valid = student_indices >= 0  # (B, S, K)
        masked_probs = teacher_probs * valid.float()
        masked_indices = student_indices.clamp(min=0)  # safe for scatter

        # Scatter add mapped probabilities
        student_dist.scatter_add_(2, masked_indices, masked_probs)

        # Redistribute unmapped mass uniformly
        mapped_mass = masked_probs.sum(dim=-1, keepdim=True)  # (B, S, 1)
        unmapped_mass = (1.0 - mapped_mass).clamp(min=0)
        student_dist += unmapped_mass / student_vocab_size

        # Normalize
        student_dist = student_dist / student_dist.sum(dim=-1, keepdim=True)

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
        with open(index_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                # Verify NPZ exists
                npz_path = self.logits_dir / entry["logits_file"]
                if npz_path.exists():
                    self.samples.append(entry)

        logger.info(f"Loaded {len(self.samples)} samples (with valid NPZ files)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load logits
        logits_file = self.logits_dir / sample["logits_file"]
        data = np.load(logits_file)

        teacher_probs = torch.from_numpy(data["top_k_probs"].astype(np.float32))
        teacher_indices = torch.from_numpy(data["top_k_indices"].astype(np.int64))
        n_response_tokens = teacher_probs.shape[0]

        # Format input for student
        input_text = sample["input_text"]
        response_text = sample["response_text"]

        prompt = (
            f"Analyze the following financial text and provide a structured assessment:\n\n"
            f"Input: {input_text}\n\nAnalysis:\n{response_text}"
        )

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
            "teacher_probs": teacher_probs,       # (n_response_tokens, 100)
            "teacher_indices": teacher_indices,    # (n_response_tokens, 100)
            "n_response_tokens": n_response_tokens,
        }


def kd_collate_fn(batch):
    """Custom collate that pads teacher logits to the same sequence length."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])

    # Pad teacher probs/indices to max response length in this batch
    max_resp = max(b["n_response_tokens"] for b in batch)
    top_k = batch[0]["teacher_probs"].shape[1]

    teacher_probs = torch.zeros(len(batch), max_resp, top_k)
    teacher_indices = torch.zeros(len(batch), max_resp, top_k, dtype=torch.long)
    resp_mask = torch.zeros(len(batch), max_resp, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b["n_response_tokens"]
        teacher_probs[i, :n] = b["teacher_probs"]
        teacher_indices[i, :n] = b["teacher_indices"]
        resp_mask[i, :n] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "teacher_probs": teacher_probs,
        "teacher_indices": teacher_indices,
        "resp_mask": resp_mask,
    }


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
    L = alpha * L_hard + (1-alpha) * T^2 * L_soft

    Where:
    - L_hard: Cross-entropy on hard labels (teacher's generated tokens)
    - L_soft: KL divergence on soft labels (teacher's probability distribution)
    - T: Temperature
    - alpha: Mixing coefficient
    """

    def __init__(self, alpha: float = ALPHA, temperature: float = TEMPERATURE):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, labels: torch.Tensor,
                teacher_soft_labels: torch.Tensor = None,
                soft_mask: torch.Tensor = None) -> dict:
        """
        Args:
            student_logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len) hard labels
            teacher_soft_labels: (batch, resp_len, vocab_size) soft labels
            soft_mask: (batch, resp_len) bool mask for valid soft label positions

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
                "soft_loss": torch.tensor(0.0, device=student_logits.device)
            }

        # Soft label loss — only on response token positions
        # We apply KL on the last `resp_len` positions of student logits
        resp_len = teacher_soft_labels.shape[1]
        # Take the last resp_len positions of student logits (response region)
        student_resp_logits = student_logits[:, -resp_len:, :]

        # Apply temperature
        student_soft = F.log_softmax(student_resp_logits / self.temperature, dim=-1)

        # KL divergence per position
        kl_per_pos = F.kl_div(
            student_soft,
            teacher_soft_labels,
            reduction='none',
            log_target=False
        ).sum(dim=-1)  # (B, resp_len)

        # Apply mask (ignore padded response positions)
        if soft_mask is not None:
            kl_per_pos = kl_per_pos * soft_mask.float()
            soft_loss = kl_per_pos.sum() / soft_mask.float().sum().clamp(min=1.0)
        else:
            soft_loss = kl_per_pos.mean()

        # Scale by T^2
        soft_loss = soft_loss * (self.temperature ** 2)

        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return {
            "total_loss": total_loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss
        }


def train_epoch(model, dataloader, optimizer, scheduler, loss_fn,
                vocab_projector, student_vocab_size, device, epoch,
                grad_accum_steps=1, use_soft_labels=False):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    total_hard = 0
    total_soft = 0
    n_steps = 0

    optimizer.zero_grad()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]")

    for step, batch in enumerate(pbar):
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
        # Mask padding positions
        pad_mask = attention_mask[:, 1:] == 0
        labels[pad_mask] = -100

        # Project teacher distributions if available
        teacher_soft = None
        soft_mask = None
        if use_soft_labels and vocab_projector is not None:
            teacher_probs = batch["teacher_probs"].to(device)
            teacher_indices = batch["teacher_indices"].to(device)
            resp_mask = batch["resp_mask"].to(device)

            teacher_soft = vocab_projector.project_distribution_batched(
                teacher_probs, teacher_indices, student_vocab_size
            )
            soft_mask = resp_mask

        # Compute loss
        losses = loss_fn(logits, labels, teacher_soft, soft_mask)
        loss = losses["total_loss"] / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += losses["total_loss"].item()
        total_hard += losses["hard_loss"].item()
        total_soft += losses["soft_loss"].item()
        n_steps += 1

        pbar.set_postfix({
            "loss": f"{losses['total_loss'].item():.4f}",
            "hard": f"{losses['hard_loss'].item():.4f}",
            "soft": f"{losses['soft_loss'].item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })

    return total_loss / n_steps, total_hard / n_steps, total_soft / n_steps


@torch.no_grad()
def validate(model, dataloader, loss_fn, vocab_projector,
             student_vocab_size, device, use_soft_labels=False):
    """Validation loop"""
    model.eval()
    total_loss = 0
    total_hard = 0
    total_soft = 0
    n_steps = 0

    for batch in tqdm(dataloader, desc="Validating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_ids[:, 1:].contiguous()
        pad_mask = attention_mask[:, 1:] == 0
        labels[pad_mask] = -100

        teacher_soft = None
        soft_mask = None
        if use_soft_labels and vocab_projector is not None:
            teacher_probs = batch["teacher_probs"].to(device)
            teacher_indices = batch["teacher_indices"].to(device)
            resp_mask = batch["resp_mask"].to(device)
            teacher_soft = vocab_projector.project_distribution_batched(
                teacher_probs, teacher_indices, student_vocab_size
            )
            soft_mask = resp_mask

        losses = loss_fn(logits, labels, teacher_soft, soft_mask)
        total_loss += losses["total_loss"].item()
        total_hard += losses["hard_loss"].item()
        total_soft += losses["soft_loss"].item()
        n_steps += 1

    return total_loss / n_steps, total_hard / n_steps, total_soft / n_steps


def main():
    parser = argparse.ArgumentParser(description="Train with Knowledge Distillation")
    parser.add_argument("--student-model", type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument("--logits-dir", type=str, default=DEFAULT_LOGITS_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help="Weight for hard loss (0=pure KD, 1=pure CE)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.05,
                        help="Fraction of data for validation")
    parser.add_argument("--use-soft-labels", action="store_true",
                        help="Use soft labels from teacher (requires vocab projection)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Save checkpoint every N steps (0 = only at epoch end)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load student model and tokenizer
    logger.info(f"Loading student model: {args.student_model}")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    student_tokenizer.pad_token = student_tokenizer.eos_token
    student_vocab_size = len(student_tokenizer)

    model = MambaForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=torch.bfloat16
    ).to(device)

    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: enabled")

    # Check Mamba fast kernels
    try:
        import mamba_ssm  # noqa: F401
        logger.info("Mamba fast kernels: AVAILABLE")
    except ImportError:
        logger.warning(
            "Mamba fast kernels NOT installed — falling back to slow path. "
            "Install with: pip install causal-conv1d>=1.2.0 mamba-ssm>=1.2.0"
        )

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
            logger.warning("Teacher tokenizer not found, falling back to hard labels only")
            args.alpha = 1.0
            args.use_soft_labels = False

    # Create dataset
    full_dataset = KDDataset(
        args.logits_dir,
        student_tokenizer,
        max_length=args.max_length
    )

    # Train/val split
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Split: {n_train} train / {n_val} val (total {n_total})")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        collate_fn=kd_collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        collate_fn=kd_collate_fn
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Loss function
    loss_fn = DistillationLoss(alpha=args.alpha, temperature=args.temperature)

    # Resume from checkpoint
    start_epoch = 1
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt_state = os.path.join(args.resume, "training_state.pt")
        if os.path.exists(ckpt_state):
            state = torch.load(ckpt_state, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"] + 1
            best_val_loss = state.get("best_val_loss", float('inf'))
            logger.info(f"Resumed from epoch {state['epoch']}, best_val_loss={best_val_loss:.4f}")
        else:
            logger.info(f"Loading model weights from {args.resume}")
            model = MambaForCausalLM.from_pretrained(
                args.resume, torch_dtype=torch.bfloat16
            ).to(device)

    # Training loop
    logger.info("=" * 60)
    logger.info("Starting Knowledge Distillation training")
    logger.info(f"  Dataset:         {n_train} train / {n_val} val")
    logger.info(f"  Batch size:      {args.batch_size} x {args.grad_accum} accum = {args.batch_size * args.grad_accum} effective")
    logger.info(f"  Epochs:          {args.epochs}")
    logger.info(f"  Total steps:     {total_steps}")
    logger.info(f"  Warmup steps:    {warmup_steps}")
    logger.info(f"  Learning rate:   {args.lr}")
    logger.info(f"  Alpha (hard):    {args.alpha}")
    logger.info(f"  Temperature:     {args.temperature}")
    logger.info(f"  Soft labels:     {args.use_soft_labels}")
    logger.info(f"  Max seq length:  {args.max_length}")
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss, train_hard, train_soft = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, vocab_projector, student_vocab_size, device, epoch,
            grad_accum_steps=args.grad_accum,
            use_soft_labels=args.use_soft_labels
        )

        # Validate
        val_loss, val_hard, val_soft = validate(
            model, val_loader, loss_fn, vocab_projector,
            student_vocab_size, device,
            use_soft_labels=args.use_soft_labels
        )

        logger.info(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f} (hard={train_hard:.4f}, soft={train_soft:.4f}) | "
            f"val_loss={val_loss:.4f} (hard={val_hard:.4f}, soft={val_soft:.4f})"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            student_tokenizer.save_pretrained(best_dir)
            # Save full training state for resume
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, os.path.join(best_dir, "training_state.pt"))
            logger.info(f"  -> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"  -> No improvement ({patience_counter}/{args.patience})")

        # Save epoch checkpoint
        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break

    # Save final model
    logger.info(f"Saving final model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    student_tokenizer.save_pretrained(args.output_dir)

    # Save training config
    config = {
        "student_model": args.student_model,
        "epochs": epoch,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "use_soft_labels": args.use_soft_labels,
        "best_val_loss": best_val_loss,
        "n_train": n_train,
        "n_val": n_val,
        "max_length": args.max_length,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
    }
    with open(os.path.join(args.output_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
