"""
SOTA Causal Predictor: News Embeddings → Market Outcomes

This script trains a model to learn the causal relationship between
news embeddings and market outcomes (returns, volatility, regime changes).

Architecture: Cross-Attention Transformer with Temporal Encoding
- Input: News embeddings (batch of embeddings per day/ticker)
- Output: Multi-task predictions (returns, volatility, direction)

Key Features:
1. Temporal positional encoding for time-aware predictions
2. Cross-attention between news and market context
3. Multi-task heads for comprehensive market modeling
4. Contrastive learning for better representation

Usage on RunPod (A100 recommended):
    python scripts/train_causal_predictor.py \
        --embeddings_path /workspace/embeddings/news_embeddings.npz \
        --market_data_path /workspace/data/eodhd_sp500.db \
        --output_dir /workspace/models/causal_predictor \
        --epochs 50 \
        --batch_size 64
"""

import argparse
import json
import math
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from loguru import logger

# Try to import optional dependencies
try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


# ============================================================================
# Model Architecture
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal awareness."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if positions is not None:
            # Custom positions (e.g., day of week, hour, etc.)
            pe = self.pe[:, positions, :]
        else:
            pe = self.pe[:, :x.size(1), :]
        return self.dropout(x + pe)


class NewsAggregator(nn.Module):
    """
    Aggregates multiple news embeddings per (date, ticker) pair.
    Uses attention to weight importance of each news item.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Learnable query for aggregation
        self.agg_query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, num_news, embed_dim)
            mask: (batch, num_news) - True for padding

        Returns:
            (batch, embed_dim) aggregated embedding
        """
        batch_size = embeddings.size(0)
        query = self.agg_query.expand(batch_size, -1, -1)

        attn_out, _ = self.attention(
            query, embeddings, embeddings,
            key_padding_mask=mask
        )

        return self.norm(attn_out.squeeze(1))


class MarketContextEncoder(nn.Module):
    """
    Encodes historical market context (past returns, volatility, regime).
    """

    def __init__(self, context_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.encoder(context)


class CrossAttentionBlock(nn.Module):
    """Cross-attention between news and market context."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, news: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross-attention: news attends to context
        attn_out, _ = self.cross_attn(news.unsqueeze(1), context.unsqueeze(1), context.unsqueeze(1))
        news = self.norm1(news + attn_out.squeeze(1))

        # FFN
        news = self.norm2(news + self.ffn(news))
        return news


class CausalPredictor(nn.Module):
    """
    SOTA Causal Predictor: News → Market Outcomes

    Architecture:
    1. News Aggregator (attention over multiple news per day)
    2. Market Context Encoder (past returns, volatility, regime)
    3. Cross-Attention Fusion
    4. Multi-task Prediction Heads
    """

    def __init__(
        self,
        embed_dim: int = 4096,  # Qwen3 embedding dimension
        hidden_dim: int = 512,
        context_dim: int = 32,  # Historical market features
        num_heads: int = 8,
        num_cross_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        # Project embedding dimension down
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # News aggregator
        self.news_aggregator = NewsAggregator(hidden_dim, num_heads)

        # Market context encoder
        self.context_encoder = MarketContextEncoder(context_dim, hidden_dim, hidden_dim)

        # Cross-attention layers
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_cross_layers)
        ])

        # Multi-task prediction heads
        self.return_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Continuous return
        )

        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # Up, Down, Neutral
        )

        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Log volatility
        )

        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Bull, Bear
        )

        # Contrastive projection head
        self.contrastive_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128)  # Lower dim for contrastive
        )

    def forward(
        self,
        news_embeddings: torch.Tensor,
        market_context: torch.Tensor,
        news_mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            news_embeddings: (batch, num_news, embed_dim)
            market_context: (batch, context_dim) - historical features
            news_mask: (batch, num_news) - True for padding
            return_features: If True, also return intermediate features

        Returns:
            Dictionary with predictions for each task
        """
        # Project embeddings
        news = self.input_proj(news_embeddings)  # (batch, num_news, hidden)

        # Aggregate news
        news_agg = self.news_aggregator(news, news_mask)  # (batch, hidden)

        # Encode market context
        context = self.context_encoder(market_context)  # (batch, hidden)

        # Cross-attention fusion
        fused = news_agg
        for cross_layer in self.cross_layers:
            fused = cross_layer(fused, context)

        # Predictions
        outputs = {
            "return_pred": self.return_head(fused).squeeze(-1),
            "direction_logits": self.direction_head(fused),
            "volatility_pred": self.volatility_head(fused).squeeze(-1),
            "regime_logits": self.regime_head(fused)
        }

        if return_features:
            outputs["features"] = fused
            outputs["contrastive"] = F.normalize(self.contrastive_proj(fused), dim=-1)

        return outputs


# ============================================================================
# Dataset
# ============================================================================

class CausalDataset(Dataset):
    """
    Dataset for causal prediction: news embeddings → market outcomes.
    """

    def __init__(
        self,
        embeddings_path: str,
        db_path: str,
        split: str = "train",
        max_news_per_sample: int = 10,
        lookback_days: int = 5,
        forecast_horizon: int = 1,
        train_ratio: float = 0.8
    ):
        self.max_news = max_news_per_sample
        self.lookback = lookback_days
        self.horizon = forecast_horizon

        # Load embeddings
        logger.info(f"Loading embeddings from {embeddings_path}...")
        data = np.load(embeddings_path, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.embed_dim = self.embeddings.shape[1]

        # Parse metadata
        self.metadata = [json.loads(m) for m in data['metadata']]

        # Build index: (date, ticker) -> list of embedding indices
        self.index = {}
        for i, m in enumerate(self.metadata):
            key = (m['date'], m['ticker'])
            if key not in self.index:
                self.index[key] = []
            self.index[key].append(i)

        # Load market data from database
        logger.info(f"Loading market data from {db_path}...")
        self.market_data = self._load_market_data(db_path)

        # Build samples: each sample is (date, ticker) with news and market data
        self.samples = self._build_samples()

        # Train/test split
        n_train = int(len(self.samples) * train_ratio)
        if split == "train":
            self.samples = self.samples[:n_train]
        else:
            self.samples = self.samples[n_train:]

        logger.info(f"Created {len(self.samples)} {split} samples")

    def _load_market_data(self, db_path: str) -> Dict:
        """Load price data from database."""
        conn = sqlite3.connect(db_path)

        query = """
            SELECT ticker, date, adj_close, volume
            FROM prices
            ORDER BY ticker, date
        """
        cursor = conn.execute(query)

        data = {}
        for row in cursor:
            ticker, date, close, volume = row
            if ticker not in data:
                data[ticker] = {'dates': [], 'prices': [], 'volumes': []}
            data[ticker]['dates'].append(date)
            data[ticker]['prices'].append(close)
            data[ticker]['volumes'].append(volume)

        conn.close()

        # Convert to numpy for faster access
        for ticker in data:
            data[ticker]['prices'] = np.array(data[ticker]['prices'], dtype=np.float32)
            data[ticker]['volumes'] = np.array(data[ticker]['volumes'], dtype=np.float32)
            data[ticker]['date_idx'] = {d: i for i, d in enumerate(data[ticker]['dates'])}

        return data

    def _build_samples(self) -> List[Tuple]:
        """Build list of valid (date, ticker) samples."""
        samples = []

        for (date, ticker), emb_indices in self.index.items():
            if ticker not in self.market_data:
                continue

            date_idx = self.market_data[ticker]['date_idx'].get(date)
            if date_idx is None:
                continue

            # Need enough history and future
            if date_idx < self.lookback or date_idx + self.horizon >= len(self.market_data[ticker]['prices']):
                continue

            samples.append((date, ticker, emb_indices, date_idx))

        # Sort by date for proper train/test split
        samples.sort(key=lambda x: x[0])
        return samples

    def _compute_market_context(self, ticker: str, date_idx: int) -> np.ndarray:
        """Compute historical market features."""
        prices = self.market_data[ticker]['prices']
        volumes = self.market_data[ticker]['volumes']

        # Lookback window
        hist_prices = prices[date_idx - self.lookback:date_idx + 1]
        hist_volumes = volumes[date_idx - self.lookback:date_idx + 1]

        # Returns
        returns = np.diff(hist_prices) / hist_prices[:-1]

        # Volatility (std of returns)
        volatility = np.std(returns)

        # Momentum (cumulative return)
        momentum = (hist_prices[-1] / hist_prices[0]) - 1

        # Volume trend
        vol_trend = (hist_volumes[-1] / hist_volumes[0]) - 1 if hist_volumes[0] > 0 else 0

        # Features: past returns + stats
        context = np.concatenate([
            returns,  # lookback returns
            [volatility, momentum, vol_trend],  # aggregate stats
            [np.mean(returns), np.median(returns), np.max(returns), np.min(returns)]  # return stats
        ])

        # Pad/truncate to fixed size
        context_dim = 32  # Fixed context dimension
        if len(context) < context_dim:
            context = np.pad(context, (0, context_dim - len(context)))
        else:
            context = context[:context_dim]

        return context.astype(np.float32)

    def _compute_targets(self, ticker: str, date_idx: int) -> Dict:
        """Compute prediction targets."""
        prices = self.market_data[ticker]['prices']

        # Future return
        current_price = prices[date_idx]
        future_price = prices[date_idx + self.horizon]
        future_return = (future_price / current_price) - 1

        # Direction (3 classes: up > 0.5%, down < -0.5%, neutral)
        if future_return > 0.005:
            direction = 0  # Up
        elif future_return < -0.005:
            direction = 1  # Down
        else:
            direction = 2  # Neutral

        # Volatility (realized, next period)
        if date_idx + self.horizon + 5 < len(prices):
            future_prices = prices[date_idx + 1:date_idx + self.horizon + 6]
            future_returns = np.diff(future_prices) / future_prices[:-1]
            future_vol = np.log(np.std(future_returns) + 1e-8)
        else:
            future_vol = 0.0

        # Regime (simplified: bull if cumulative return > 0 over next 5 days)
        if date_idx + 5 < len(prices):
            cum_return = (prices[date_idx + 5] / prices[date_idx]) - 1
            regime = 0 if cum_return > 0 else 1
        else:
            regime = 0

        return {
            "return": np.float32(future_return),
            "direction": direction,
            "volatility": np.float32(future_vol),
            "regime": regime
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        date, ticker, emb_indices, date_idx = self.samples[idx]

        # Get embeddings (sample if too many)
        if len(emb_indices) > self.max_news:
            emb_indices = np.random.choice(emb_indices, self.max_news, replace=False)

        embeddings = self.embeddings[emb_indices]

        # Pad to max_news
        mask = np.zeros(self.max_news, dtype=bool)
        if len(embeddings) < self.max_news:
            pad_len = self.max_news - len(embeddings)
            embeddings = np.pad(embeddings, ((0, pad_len), (0, 0)))
            mask[len(emb_indices):] = True

        # Get market context
        context = self._compute_market_context(ticker, date_idx)

        # Get targets
        targets = self._compute_targets(ticker, date_idx)

        return {
            "embeddings": torch.from_numpy(embeddings),
            "mask": torch.from_numpy(mask),
            "context": torch.from_numpy(context),
            "return_target": targets["return"],
            "direction_target": targets["direction"],
            "volatility_target": targets["volatility"],
            "regime_target": targets["regime"]
        }


# ============================================================================
# Training
# ============================================================================

class MultiTaskLoss(nn.Module):
    """Multi-task loss with learnable weights."""

    def __init__(self):
        super().__init__()
        # Log-variances (learnable)
        self.log_var_return = nn.Parameter(torch.zeros(1))
        self.log_var_direction = nn.Parameter(torch.zeros(1))
        self.log_var_volatility = nn.Parameter(torch.zeros(1))
        self.log_var_regime = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        return_pred: torch.Tensor,
        return_target: torch.Tensor,
        direction_logits: torch.Tensor,
        direction_target: torch.Tensor,
        volatility_pred: torch.Tensor,
        volatility_target: torch.Tensor,
        regime_logits: torch.Tensor,
        regime_target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Return loss (MSE)
        return_loss = F.mse_loss(return_pred, return_target)

        # Direction loss (CE)
        direction_loss = F.cross_entropy(direction_logits, direction_target)

        # Volatility loss (MSE)
        volatility_loss = F.mse_loss(volatility_pred, volatility_target)

        # Regime loss (CE)
        regime_loss = F.cross_entropy(regime_logits, regime_target)

        # Uncertainty-weighted combination
        total = (
            torch.exp(-self.log_var_return) * return_loss + self.log_var_return +
            torch.exp(-self.log_var_direction) * direction_loss + self.log_var_direction +
            torch.exp(-self.log_var_volatility) * volatility_loss + self.log_var_volatility +
            torch.exp(-self.log_var_regime) * regime_loss + self.log_var_regime
        )

        return {
            "total": total,
            "return": return_loss,
            "direction": direction_loss,
            "volatility": volatility_loss,
            "regime": regime_loss
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    metrics = {"total": 0, "return": 0, "direction": 0, "volatility": 0, "regime": 0}
    direction_correct = 0
    regime_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        embeddings = batch["embeddings"].to(device)
        mask = batch["mask"].to(device)
        context = batch["context"].to(device)
        return_target = batch["return_target"].to(device)
        direction_target = batch["direction_target"].to(device)
        volatility_target = batch["volatility_target"].to(device)
        regime_target = batch["regime_target"].to(device)

        optimizer.zero_grad()

        # Forward
        if scaler is not None:
            with autocast():
                outputs = model(embeddings, context, mask)
                losses = loss_fn(
                    outputs["return_pred"], return_target,
                    outputs["direction_logits"], direction_target,
                    outputs["volatility_pred"], volatility_target,
                    outputs["regime_logits"], regime_target
                )

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(embeddings, context, mask)
            losses = loss_fn(
                outputs["return_pred"], return_target,
                outputs["direction_logits"], direction_target,
                outputs["volatility_pred"], volatility_target,
                outputs["regime_logits"], regime_target
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Metrics
        batch_size = embeddings.size(0)
        for k, v in losses.items():
            metrics[k] += v.item() * batch_size

        direction_correct += (outputs["direction_logits"].argmax(dim=-1) == direction_target).sum().item()
        regime_correct += (outputs["regime_logits"].argmax(dim=-1) == regime_target).sum().item()
        total_samples += batch_size

        pbar.set_postfix({
            "loss": f"{losses['total'].item():.4f}",
            "dir_acc": f"{100*direction_correct/total_samples:.1f}%"
        })

    # Average metrics
    for k in metrics:
        metrics[k] /= total_samples

    metrics["direction_acc"] = direction_correct / total_samples
    metrics["regime_acc"] = regime_correct / total_samples

    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    metrics = {"total": 0, "return": 0, "direction": 0, "volatility": 0, "regime": 0}
    direction_correct = 0
    regime_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            embeddings = batch["embeddings"].to(device)
            mask = batch["mask"].to(device)
            context = batch["context"].to(device)
            return_target = batch["return_target"].to(device)
            direction_target = batch["direction_target"].to(device)
            volatility_target = batch["volatility_target"].to(device)
            regime_target = batch["regime_target"].to(device)

            outputs = model(embeddings, context, mask)
            losses = loss_fn(
                outputs["return_pred"], return_target,
                outputs["direction_logits"], direction_target,
                outputs["volatility_pred"], volatility_target,
                outputs["regime_logits"], regime_target
            )

            batch_size = embeddings.size(0)
            for k, v in losses.items():
                metrics[k] += v.item() * batch_size

            direction_correct += (outputs["direction_logits"].argmax(dim=-1) == direction_target).sum().item()
            regime_correct += (outputs["regime_logits"].argmax(dim=-1) == regime_target).sum().item()
            total_samples += batch_size

    for k in metrics:
        metrics[k] /= total_samples

    metrics["direction_acc"] = direction_correct / total_samples
    metrics["regime_acc"] = regime_correct / total_samples

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Causal Predictor")
    parser.add_argument("--embeddings_path", type=str, required=True,
                        help="Path to news embeddings .npz file")
    parser.add_argument("--market_data_path", type=str, required=True,
                        help="Path to market database")
    parser.add_argument("--output_dir", type=str, default="./models/causal_predictor",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_cross_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_news", type=int, default=10)
    parser.add_argument("--lookback_days", type=int, default=5)
    parser.add_argument("--forecast_horizon", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 60)
    logger.info("Causal Predictor Training")
    logger.info("=" * 60)
    logger.info(f"Embeddings: {args.embeddings_path}")
    logger.info(f"Market data: {args.market_data_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info("")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = CausalDataset(
        args.embeddings_path,
        args.market_data_path,
        split="train",
        max_news_per_sample=args.max_news,
        lookback_days=args.lookback_days,
        forecast_horizon=args.forecast_horizon
    )

    val_dataset = CausalDataset(
        args.embeddings_path,
        args.market_data_path,
        split="val",
        max_news_per_sample=args.max_news,
        lookback_days=args.lookback_days,
        forecast_horizon=args.forecast_horizon
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    logger.info("Creating model...")
    model = CausalPredictor(
        embed_dim=train_dataset.embed_dim,
        hidden_dim=args.hidden_dim,
        context_dim=32,
        num_heads=args.num_heads,
        num_cross_layers=args.num_cross_layers,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = MultiTaskLoss().to(device)

    # Mixed precision
    scaler = GradScaler() if args.fp16 and AMP_AVAILABLE else None

    # Training loop
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler)

        # Validate
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        scheduler.step()

        # Log
        logger.info(f"Train: loss={train_metrics['total']:.4f}, dir_acc={100*train_metrics['direction_acc']:.1f}%")
        logger.info(f"Val:   loss={val_metrics['total']:.4f}, dir_acc={100*val_metrics['direction_acc']:.1f}%")

        # Save history
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics
        })

        # Save best model
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            logger.info(f"New best! Saving model (loss={best_val_loss:.4f})")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
                "config": vars(args)
            }, output_dir / "best_model.pt")

        # Checkpoint
        if epoch % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_metrics['total'],
                "config": vars(args)
            }, output_dir / f"checkpoint_epoch_{epoch}.pt")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(args)
    }, output_dir / "final_model.pt")

    # Save history
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
