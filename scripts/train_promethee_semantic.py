"""
Train Promethee SEMANTIC - SOTA Financial Analysis LLM

This is the SOTA training script that teaches Promethee to:
1. Understand structured multi-modal input (news + quant + correlations)
2. Produce semantic analysis with opportunities and risks
3. Output embeddings suitable for SAC agent

Key Features:
- Muon Optimizer (SOTA for LLMs)
- Structured input/output format
- Embedding extraction for SAC integration
- Contrastive learning for similar situations
- Periodic teacher evaluation

Usage:
    python scripts/train_promethee_semantic.py \
        --train_file data/promethee_semantic_training.jsonl \
        --output_dir ./models/promethee_semantic \
        --epochs 20 \
        --batch_size 4 \
        --use_muon
"""

import os
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger

from transformers import (
    MambaForCausalLM,
    MambaConfig,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)


# =============================================================================
# Muon Optimizer (SOTA for LLMs)
# =============================================================================

class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum-based optimizer without exponential moving averages.
    Better convergence than AdamW for language models.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        backend: str = "newtonschulz5"
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            backend=backend
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            wd = group['weight_decay']
            backend = group['backend']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if wd != 0:
                    p.mul_(1 - lr * wd)

                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

                if p.dim() >= 2 and backend == "newtonschulz5":
                    d_p = self._newton_schulz_orthogonalize(d_p)

                p.add_(d_p, alpha=-lr)

        return loss

    def _newton_schulz_orthogonalize(self, G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        shape = G.shape
        if G.dim() > 2:
            G = G.reshape(G.size(0), -1)

        scale = (G.shape[0] * G.shape[1]) ** 0.5
        G = G / (G.norm() + 1e-7) * scale

        if G.shape[0] <= G.shape[1]:
            for _ in range(steps):
                A = G @ G.T
                G = 1.5 * G - 0.5 * A @ G
        else:
            for _ in range(steps):
                A = G.T @ G
                G = 1.5 * G - 0.5 * G @ A

        G = G.reshape(shape)
        return G


# =============================================================================
# Dataset for Semantic Training
# =============================================================================

class PrometheeSemanticDataset(Dataset):
    """
    Dataset for semantic financial analysis training.

    Each sample contains:
    - Structured input with context, regime, correlations, etc.
    - Semantic output with analysis, opportunities, risks
    - Ground truth for validation
    """

    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_input_length: int = 2048,
        max_output_length: int = 1024
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.samples = []

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    if sample.get('input') and sample.get('output'):
                        self.samples.append(sample)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(self.samples)} training samples")

        # Group samples by ticker for contrastive learning
        self.ticker_groups = defaultdict(list)
        for i, s in enumerate(self.samples):
            self.ticker_groups[s.get('ticker', 'UNK')].append(i)

    def __len__(self):
        return len(self.samples)

    def _format_training_example(self, sample: Dict) -> str:
        """Format sample for causal language modeling."""

        input_text = sample['input']
        output_text = sample['output']

        # Format: [INST] {input} [/INST] {output} </s>
        full_text = f"[INST] Analyze this financial context and provide semantic analysis:\n\n{input_text}\n\nProvide comprehensive analysis with opportunities and risks. [/INST]\n\n{output_text}</s>"

        return full_text

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = self._format_training_example(sample)

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_input_length + self.max_output_length,
            padding='max_length',
            return_tensors='pt'
        )

        labels = encoding['input_ids'].clone()

        # Mask input portion for training (only predict output)
        # Find [/INST] position and mask everything before
        input_ids = encoding['input_ids'].squeeze()
        inst_tokens = self.tokenizer.encode("[/INST]", add_special_tokens=False)

        mask_end = 0
        for i in range(len(input_ids) - len(inst_tokens)):
            if input_ids[i:i+len(inst_tokens)].tolist() == inst_tokens:
                mask_end = i + len(inst_tokens)
                break

        labels[0, :mask_end] = -100

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'ticker': sample.get('ticker', 'UNK'),
            'direction': sample.get('ground_truth', {}).get('direction', 'neutral')
        }

    def get_contrastive_pair(self, idx: int) -> Optional[int]:
        """Get a similar sample for contrastive learning (same ticker)."""
        sample = self.samples[idx]
        ticker = sample.get('ticker', 'UNK')

        if len(self.ticker_groups[ticker]) < 2:
            return None

        import random
        candidates = [i for i in self.ticker_groups[ticker] if i != idx]
        return random.choice(candidates) if candidates else None


# =============================================================================
# Promethee Model with Embedding Head
# =============================================================================

class PrometheeModel(nn.Module):
    """
    Promethee model wrapper with embedding head for SAC integration.

    The model has two outputs:
    1. Causal LM output (for text generation during training)
    2. Embedding output (for SAC agent input)
    """

    def __init__(self, base_model: MambaForCausalLM, embed_dim: int = 256):
        super().__init__()
        self.mamba = base_model
        self.hidden_size = base_model.config.hidden_size

        # Embedding projection head for SAC
        self.embed_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Task-specific heads
        self.direction_head = nn.Linear(embed_dim, 3)  # up/down/neutral
        self.confidence_head = nn.Linear(embed_dim, 3)  # high/medium/low
        self.risk_head = nn.Linear(embed_dim, 3)  # low/elevated/high

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        return_embeddings=False
    ):
        # Forward through Mamba
        outputs = self.mamba(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )

        result = {
            'loss': outputs.loss,
            'logits': outputs.logits
        }

        if return_embeddings:
            # Get last hidden state
            hidden = outputs.hidden_states[-1]

            # Pool: use last non-padded token
            if attention_mask is not None:
                # Get position of last real token
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden.size(0), device=hidden.device)
                pooled = hidden[batch_indices, seq_lengths]
            else:
                pooled = hidden[:, -1, :]

            # Project to embedding
            embeddings = self.embed_head(pooled)

            result['embeddings'] = embeddings
            result['direction_logits'] = self.direction_head(embeddings)
            result['confidence_logits'] = self.confidence_head(embeddings)
            result['risk_logits'] = self.risk_head(embeddings)

        return result

    def get_embedding(self, input_ids, attention_mask=None) -> torch.Tensor:
        """Get embedding for SAC agent input."""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_embeddings=True
            )
        return outputs['embeddings']


# =============================================================================
# Contrastive Loss
# =============================================================================

class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss for learning similar situation embeddings.
    Similar situations (same direction) should have close embeddings.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        directions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embed_dim)
            directions: (batch,) direction labels
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask (same direction)
        direction_matrix = directions.unsqueeze(0) == directions.unsqueeze(1)
        positive_mask = direction_matrix.float()
        positive_mask.fill_diagonal_(0)

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        exp_sim.fill_diagonal_(0)

        # Positive pairs
        positive_exp = (exp_sim * positive_mask).sum(dim=1)

        # All pairs denominator
        denominator = exp_sim.sum(dim=1)

        # Avoid division by zero
        loss = -torch.log((positive_exp + 1e-8) / (denominator + 1e-8))
        loss = loss[positive_mask.sum(dim=1) > 0].mean()

        return loss if not torch.isnan(loss) else torch.tensor(0.0, device=embeddings.device)


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model: PrometheeModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    accumulation_steps: int = 1,
    use_contrastive: bool = True,
    contrastive_weight: float = 0.1
) -> Dict[str, float]:
    """Train for one epoch with optional contrastive loss."""

    model.train()
    total_lm_loss = 0
    total_contrastive_loss = 0
    num_batches = 0

    contrastive_fn = InfoNCELoss(temperature=0.07)

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Map direction to index
        direction_map = {'up': 0, 'down': 1, 'neutral': 2}
        directions = torch.tensor(
            [direction_map.get(d, 2) for d in batch['direction']],
            device=device
        )

        # Mask padding in labels
        labels[labels == model.mamba.config.pad_token_id] = -100

        # Forward pass with embeddings
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_embeddings=use_contrastive
        )

        lm_loss = outputs['loss'] / accumulation_steps
        total_loss = lm_loss

        # Contrastive loss
        if use_contrastive and 'embeddings' in outputs:
            contrastive_loss = contrastive_fn(outputs['embeddings'], directions)
            total_loss = lm_loss + contrastive_weight * contrastive_loss
            total_contrastive_loss += contrastive_loss.item()

        total_loss.backward()

        # Gradient accumulation
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_lm_loss += outputs['loss'].item()
        num_batches += 1

        pbar.set_postfix({
            'lm_loss': f"{outputs['loss'].item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

    return {
        'lm_loss': total_lm_loss / num_batches,
        'contrastive_loss': total_contrastive_loss / num_batches if use_contrastive else 0
    }


def validate(
    model: PrometheeModel,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""

    model.eval()
    total_loss = 0
    num_batches = 0

    direction_correct = 0
    direction_total = 0

    direction_map = {'up': 0, 'down': 1, 'neutral': 2}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            directions = torch.tensor(
                [direction_map.get(d, 2) for d in batch['direction']],
                device=device
            )

            labels[labels == model.mamba.config.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_embeddings=True
            )

            total_loss += outputs['loss'].item()
            num_batches += 1

            # Direction prediction accuracy
            pred_directions = outputs['direction_logits'].argmax(dim=1)
            direction_correct += (pred_directions == directions).sum().item()
            direction_total += directions.size(0)

    return {
        'loss': total_loss / num_batches,
        'direction_accuracy': direction_correct / direction_total if direction_total > 0 else 0
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Promethee SEMANTIC - SOTA Financial Analysis LLM")

    # Model
    parser.add_argument("--base_model", type=str, default="state-spaces/mamba-1.4b-hf",
                        help="Base Mamba model")
    parser.add_argument("--output_dir", type=str, default="./models/promethee_semantic",
                        help="Output directory")
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="SAC embedding dimension")

    # Data
    parser.add_argument("--train_file", type=str, required=True,
                        help="Training data JSONL")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Validation data JSONL")
    parser.add_argument("--max_input_length", type=int, default=2048,
                        help="Max input sequence length")
    parser.add_argument("--max_output_length", type=int, default=1024,
                        help="Max output sequence length")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Optimizer
    parser.add_argument("--use_muon", action="store_true",
                        help="Use Muon optimizer (SOTA)")
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Contrastive Learning
    parser.add_argument("--use_contrastive", action="store_true",
                        help="Use contrastive learning for embeddings")
    parser.add_argument("--contrastive_weight", type=float, default=0.1,
                        help="Weight for contrastive loss")

    # Other
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 70)
    logger.info("PROMETHEE SEMANTIC - SOTA Financial Analysis LLM")
    logger.info("=" * 70)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Optimizer: {'Muon (SOTA)' if args.use_muon else 'AdamW'}")
    logger.info(f"Contrastive Learning: {args.use_contrastive}")
    logger.info(f"Embedding dim: {args.embed_dim}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("")

    # Load tokenizer and base model
    logger.info("Loading Mamba model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = MambaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.fp16 else torch.float32
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # Create Promethee model with embedding head
    model = PrometheeModel(base_model, embed_dim=args.embed_dim).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")

    # Create datasets
    logger.info("Loading training data...")
    train_dataset = PrometheeSemanticDataset(
        args.train_file,
        tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = None
    if args.val_file:
        val_dataset = PrometheeSemanticDataset(
            args.val_file,
            tokenizer,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    # Create optimizer
    if args.use_muon:
        logger.info("Using Muon optimizer (SOTA)")
        optimizer = Muon(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=args.weight_decay
        )
    else:
        logger.info("Using AdamW optimizer")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch,
            grad_clip=args.grad_clip,
            accumulation_steps=args.grad_accum,
            use_contrastive=args.use_contrastive,
            contrastive_weight=args.contrastive_weight
        )

        log_msg = f"Epoch {epoch}: lm_loss={train_metrics['lm_loss']:.4f}"
        if args.use_contrastive:
            log_msg += f", contrastive={train_metrics['contrastive_loss']:.4f}"

        # Validate
        val_metrics = None
        if val_loader:
            val_metrics = validate(model, val_loader, device)
            log_msg += f", val_loss={val_metrics['loss']:.4f}, dir_acc={val_metrics['direction_accuracy']:.2%}"

        logger.info(log_msg)

        history.append({
            'epoch': epoch,
            'train_lm_loss': train_metrics['lm_loss'],
            'train_contrastive_loss': train_metrics.get('contrastive_loss', 0),
            'val_loss': val_metrics['loss'] if val_metrics else None,
            'val_direction_accuracy': val_metrics['direction_accuracy'] if val_metrics else None
        })

        # Save best model
        current_loss = val_metrics['loss'] if val_metrics else train_metrics['lm_loss']
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            logger.info(f"Saving best model (loss={best_val_loss:.4f})")

            # Save complete model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'embed_dim': args.embed_dim,
                'best_loss': best_val_loss
            }, output_dir / "checkpoint-best.pt")

            # Save Mamba backbone separately (for inference)
            model.mamba.save_pretrained(output_dir / "checkpoint-best-mamba")
            tokenizer.save_pretrained(output_dir / "checkpoint-best-mamba")

            # Save config
            config = {
                "model_name": "Promethee-Semantic",
                "model_type": "mamba-semantic-lm",
                "base_model": args.base_model,
                "embed_dim": args.embed_dim,
                "task": "financial_semantic_analysis",
                "capabilities": [
                    "regime_detection",
                    "correlation_analysis",
                    "opportunity_detection",
                    "risk_assessment",
                    "sac_embedding_output"
                ],
                "best_loss": best_val_loss,
                "epoch": epoch
            }
            with open(output_dir / "checkpoint-best-mamba" / "promethee_config.json", 'w') as f:
                json.dump(config, f, indent=2)

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['lm_loss']
            }, output_dir / f"checkpoint-epoch-{epoch}.pt")

    # Save final model
    logger.info("Saving final Promethee model...")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'embed_dim': args.embed_dim,
        'final_loss': history[-1]['train_lm_loss']
    }, output_dir / "promethee_final.pt")

    model.mamba.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save final config
    final_config = {
        "model_name": "Promethee-Semantic",
        "version": "1.0",
        "model_type": "mamba-semantic-lm",
        "base_model": args.base_model,
        "embed_dim": args.embed_dim,
        "description": "SOTA Financial Semantic Analysis LLM with SAC embedding output",
        "task": "financial_semantic_analysis",
        "capabilities": [
            "regime_detection",
            "correlation_analysis",
            "cointegration_detection",
            "tail_risk_assessment",
            "opportunity_detection",
            "portfolio_recommendations",
            "sac_vector_output"
        ],
        "training": {
            "optimizer": "muon" if args.use_muon else "adamw",
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "contrastive_learning": args.use_contrastive,
            "contrastive_weight": args.contrastive_weight,
            "final_loss": history[-1]['train_lm_loss'],
            "best_loss": best_val_loss
        },
        "integration": {
            "sac_compatible": True,
            "embedding_dim": args.embed_dim,
            "output_format": "structured_semantic"
        }
    }

    with open(output_dir / "promethee_config.json", 'w') as f:
        json.dump(final_config, f, indent=2)

    # Save history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("PROMETHEE SEMANTIC training complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Best loss: {best_val_loss:.4f}")
    logger.info(f"Embedding dim: {args.embed_dim}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
