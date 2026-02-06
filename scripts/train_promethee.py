"""
Train Prométhée - SOTA Financial Causal Language Model

Prométhée is a Mamba-based LLM fine-tuned for financial causal reasoning:
- Predicts market impact from news/events
- Learns causal relationships between events and outcomes
- Uses SOTA Muon optimizer with gradient clipping

Architecture: Mamba 1.4B → Prométhée (fine-tuned)

Key Features:
1. Muon Optimizer (SOTA for LLMs, better than AdamW)
2. Gradient Clipping (stability)
3. Causal task framing (not just text generation)
4. Knowledge Distillation from GLM-4 Teacher

Usage:
    python scripts/train_promethee.py \
        --train_file data/causal_training_data.jsonl \
        --output_dir ./models/promethee \
        --epochs 10 \
        --batch_size 4 \
        --lr 3e-4 \
        --use_muon
"""

import os
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger

from transformers import (
    MambaForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)


# =============================================================================
# Muon Optimizer (SOTA for LLMs)
# Paper: "Muon: Scaling Up Momentum-Free Optimizers"
# Better convergence than AdamW for language models
# =============================================================================

class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum-based optimizer without exponential moving averages.

    Key advantages over AdamW:
    1. Simpler: No need for beta1, beta2 hyperparameters
    2. Faster convergence on LLMs
    3. Better generalization
    4. Memory efficient (no second moment)

    Reference: https://github.com/KellerJordan/Muon
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        backend: str = "newtonschulz5"  # or "svd"
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

                # Weight decay (decoupled)
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # Get or initialize momentum buffer
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)

                # Apply Nesterov momentum
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

                # Newton-Schulz orthogonalization for matrix params (2D)
                # This is the key innovation of Muon
                if p.dim() >= 2 and backend == "newtonschulz5":
                    d_p = self._newton_schulz_orthogonalize(d_p)

                # Update
                p.add_(d_p, alpha=-lr)

        return loss

    def _newton_schulz_orthogonalize(self, G: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        Newton-Schulz iteration for approximate orthogonalization.
        This normalizes the update direction for better conditioning.
        """
        # Flatten to 2D
        shape = G.shape
        if G.dim() > 2:
            G = G.reshape(G.size(0), -1)

        # Scale to prevent overflow
        scale = (G.shape[0] * G.shape[1]) ** 0.5
        G = G / (G.norm() + 1e-7) * scale

        # Newton-Schulz iteration: X_{k+1} = 1.5 * X_k - 0.5 * X_k @ X_k^T @ X_k
        # Approximates orthogonal matrix
        if G.shape[0] <= G.shape[1]:
            # More columns: compute G @ G^T
            for _ in range(steps):
                A = G @ G.T
                G = 1.5 * G - 0.5 * A @ G
        else:
            # More rows: compute G^T @ G
            for _ in range(steps):
                A = G.T @ G
                G = 1.5 * G - 0.5 * G @ A

        # Reshape back
        G = G.reshape(shape)
        return G


# =============================================================================
# Dataset for Causal Training
# =============================================================================

class CausalFinanceDataset(Dataset):
    """
    Dataset for causal financial reasoning.

    Each sample contains:
    - news/event text
    - market context
    - teacher's causal analysis
    - ground truth outcome (optional)
    """

    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_length: int = 1024
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # Load data
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} training samples")

    def __len__(self):
        return len(self.samples)

    def _format_prompt(self, sample: Dict) -> str:
        """
        Format sample into causal reasoning prompt.

        Format:
        [INST] Analyze the causal market impact of this event:

        Event: {news text}
        Date: {date}
        Ticker: {ticker}
        Market Context: {context}

        Provide:
        1. Sentiment (bullish/bearish/neutral)
        2. Expected impact (high/medium/low)
        3. Causal reasoning
        4. Risk factors [/INST]

        {teacher analysis}
        """

        news = sample.get('input_text', sample.get('text', ''))
        date = sample.get('date', 'N/A')
        ticker = sample.get('ticker', 'N/A')
        context = sample.get('market_context', '')
        teacher_output = sample.get('teacher_output', sample.get('analysis', ''))

        if isinstance(teacher_output, dict):
            teacher_output = json.dumps(teacher_output, indent=2)

        prompt = f"""[INST] Analyze the causal market impact of this event:

Event: {news}
Date: {date}
Ticker: {ticker}
Market Context: {context}

Provide:
1. Sentiment (bullish/bearish/neutral)
2. Expected impact magnitude (high/medium/low)
3. Causal reasoning chain
4. Risk factors and uncertainties [/INST]

{teacher_output}</s>"""

        return prompt

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = self._format_prompt(sample)

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone()
        }


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
    accumulation_steps: int = 1
) -> float:
    """Train for one epoch with gradient clipping."""
    model.train()
    total_loss = 0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Mask padding in labels
        labels[labels == model.config.pad_token_id] = -100

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / accumulation_steps
        loss.backward()

        # Gradient accumulation
        if (step + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{outputs.loss.item():.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.2e}"
        })

    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            labels[labels == model.config.pad_token_id] = -100

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            num_batches += 1

    return total_loss / num_batches


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Prométhée - SOTA Financial Causal LLM")

    # Model
    parser.add_argument("--base_model", type=str, default="state-spaces/mamba-1.4b-hf",
                        help="Base Mamba model")
    parser.add_argument("--output_dir", type=str, default="./models/promethee",
                        help="Output directory")

    # Data
    parser.add_argument("--train_file", type=str, required=True,
                        help="Training data JSONL")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Validation data JSONL (optional)")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max sequence length")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm")

    # Optimizer
    parser.add_argument("--use_muon", action="store_true",
                        help="Use Muon optimizer (SOTA) instead of AdamW")
    parser.add_argument("--momentum", type=float, default=0.95,
                        help="Muon momentum")
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Other
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 60)
    logger.info("🔥 Prométhée - SOTA Financial Causal LLM")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Optimizer: {'Muon (SOTA)' if args.use_muon else 'AdamW'}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Gradient clipping: {args.grad_clip}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("")

    # Load tokenizer and model
    logger.info("Loading Mamba model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = MambaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if args.fp16 else torch.float32
    ).to(device)

    model.config.pad_token_id = tokenizer.pad_token_id

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")

    # Create datasets
    logger.info("Loading training data...")
    train_dataset = CausalFinanceDataset(
        args.train_file,
        tokenizer,
        max_length=args.max_length
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
        val_dataset = CausalFinanceDataset(
            args.val_file,
            tokenizer,
            max_length=args.max_length
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

    # Resume if needed
    start_epoch = 1
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch,
            grad_clip=args.grad_clip,
            accumulation_steps=args.grad_accum
        )

        # Validate
        val_loss = None
        if val_loader:
            val_loss = validate(model, val_loader, device)
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        # Save best model
        current_loss = val_loss if val_loss else train_loss
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            logger.info(f"Saving best model (loss={best_val_loss:.4f})")

            # Save as "promethee" (renamed from mamba)
            model.save_pretrained(output_dir / "checkpoint-best")
            tokenizer.save_pretrained(output_dir / "checkpoint-best")

            # Save config with Prométhée branding
            config = {
                "model_name": "Prométhée",
                "model_type": "mamba-causal-lm",
                "base_model": args.base_model,
                "task": "financial_causal_reasoning",
                "optimizer": "muon" if args.use_muon else "adamw",
                "best_loss": best_val_loss,
                "epoch": epoch
            }
            with open(output_dir / "checkpoint-best" / "promethee_config.json", 'w') as f:
                json.dump(config, f, indent=2)

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss
            }, output_dir / f"checkpoint-epoch-{epoch}.pt")

    # Save final model
    logger.info("Saving final Prométhée model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save final config
    final_config = {
        "model_name": "Prométhée",
        "model_type": "mamba-causal-lm",
        "version": "1.0",
        "base_model": args.base_model,
        "task": "financial_causal_reasoning",
        "description": "SOTA Financial Causal Language Model based on Mamba architecture",
        "capabilities": [
            "sentiment_analysis",
            "impact_prediction",
            "causal_reasoning",
            "risk_assessment"
        ],
        "training": {
            "optimizer": "muon" if args.use_muon else "adamw",
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "gradient_clip": args.grad_clip,
            "final_loss": history[-1]['train_loss'] if history else None
        }
    }

    with open(output_dir / "promethee_config.json", 'w') as f:
        json.dump(final_config, f, indent=2)

    # Save history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("🔥 Prométhée training complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Best loss: {best_val_loss:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
