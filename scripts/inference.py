"""
Promethee Inference Pipeline

Complete inference pipeline:
1. Load Promethee model + SVM classifier
2. Build context for each asset
3. Generate embeddings
4. Rank assets using SVM score
5. Output portfolio recommendations

Usage:
    python scripts/inference.py \
        --model_path models/promethee_semantic \
        --svm_path models/promethee_svm.joblib \
        --db_path data/eodhd_sp500.db \
        --date 2024-12-01 \
        --top_k 10
"""

import argparse
import json
import sqlite3
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import joblib
from tqdm import tqdm
from loguru import logger

from transformers import AutoTokenizer, MambaForCausalLM


# =============================================================================
# Promethee Model
# =============================================================================

class PrometheeModel(nn.Module):
    """Promethee model with embedding head."""

    def __init__(self, base_model: MambaForCausalLM, embed_dim: int = 256):
        super().__init__()
        self.mamba = base_model
        self.hidden_size = base_model.config.hidden_size

        self.embed_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.direction_head = nn.Linear(embed_dim, 3)
        self.confidence_head = nn.Linear(embed_dim, 3)
        self.risk_head = nn.Linear(embed_dim, 3)

    def forward(self, input_ids, attention_mask=None, labels=None, return_embeddings=False):
        outputs = self.mamba(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )

        result = {'loss': outputs.loss if hasattr(outputs, 'loss') else None, 'logits': outputs.logits}

        if return_embeddings:
            hidden = outputs.hidden_states[-1]

            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden.size(0), device=hidden.device)
                pooled = hidden[batch_indices, seq_lengths]
            else:
                pooled = hidden[:, -1, :]

            embeddings = self.embed_head(pooled)
            result['embeddings'] = embeddings

        return result


# =============================================================================
# Context Builder
# =============================================================================

class ContextBuilder:
    """Build structured context for inference."""

    def __init__(self, db_path: str, thresholds_path: str = None):
        self.conn = sqlite3.connect(db_path)
        self.thresholds = None

        if thresholds_path and Path(thresholds_path).exists():
            with open(thresholds_path) as f:
                self.thresholds = json.load(f)

    def get_regime(self, date: str) -> Dict:
        """Get market regime for date."""
        cursor = self.conn.execute("""
            SELECT regime, vix_level, avg_correlation, dispersion, hmm_state
            FROM quant_regime
            WHERE date <= ?
            ORDER BY date DESC LIMIT 1
        """, (date,))
        row = cursor.fetchone()
        if row:
            return {
                'regime': row[0],
                'vix': row[1],
                'correlation': row[2],
                'dispersion': row[3],
                'hmm_state': row[4]
            }
        return {}

    def get_ticker_stats(self, ticker: str, date: str) -> Dict:
        """Get ticker statistics."""
        cursor = self.conn.execute("""
            SELECT momentum_20d, std_20d, skew_20d, kurt_20d, var_5pct
            FROM quant_daily_stats
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        """, (ticker, date))
        row = cursor.fetchone()
        if row:
            return {
                'momentum': row[0],
                'volatility': row[1],
                'skewness': row[2],
                'kurtosis': row[3],
                'var_5pct': row[4]
            }
        return {}

    def get_sector(self, ticker: str) -> str:
        """Get ticker sector."""
        cursor = self.conn.execute(
            "SELECT sector FROM fundamentals_general WHERE ticker = ?",
            (ticker,)
        )
        row = cursor.fetchone()
        return row[0] if row else 'Unknown'

    def get_correlations(self, ticker: str, date: str) -> List[Dict]:
        """Get pairwise correlations."""
        cursor = self.conn.execute("""
            SELECT ticker2, pearson, tail_dep_lower, is_cointegrated, spread_zscore, hedge_ratio
            FROM quant_pairwise
            WHERE ticker1 = ? AND date <= ?
            ORDER BY date DESC, ABS(pearson) DESC
            LIMIT 5
        """, (ticker, date))
        return [{
            'ticker': row[0],
            'pearson': row[1],
            'tail_dep': row[2],
            'coint': row[3],
            'zscore': row[4],
            'hedge': row[5]
        } for row in cursor.fetchall()]

    def get_latest_news(self, ticker: str, date: str) -> str:
        """Get latest news for ticker."""
        cursor = self.conn.execute("""
            SELECT title, content
            FROM news
            WHERE ticker = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        """, (ticker, date))
        row = cursor.fetchone()
        if row:
            return f"{row[0]}. {row[1][:500] if row[1] else ''}"
        return "No recent news."

    def build_context(self, ticker: str, date: str) -> str:
        """Build full structured context."""
        regime = self.get_regime(date)
        stats = self.get_ticker_stats(ticker, date)
        sector = self.get_sector(ticker)
        correlations = self.get_correlations(ticker, date)
        news = self.get_latest_news(ticker, date)

        # Determine mode
        vix = regime.get('vix', 18)
        corr = regime.get('correlation', 0.3)
        if vix and vix > 25:
            mode = "DEFENSIVE"
        elif vix and vix > 20:
            mode = "CAUTIOUS"
        elif corr and corr > 0.6:
            mode = "RISK-OFF"
        else:
            mode = "RISK-ON"

        parts = []

        # Context
        parts.append(f"<CONTEXT>")
        parts.append(f"TICKER: {ticker}")
        parts.append(f"DATE: {date}")
        parts.append(f"SECTOR: {sector}")
        parts.append(f"</CONTEXT>")

        # Regime
        parts.append(f"\n<REGIME>")
        parts.append(f"MODE: {mode}")
        parts.append(f"VIX: {vix:.1f}" if vix else "VIX: N/A")
        parts.append(f"CORRELATION: {corr:.3f}" if corr else "CORRELATION: N/A")
        parts.append(f"</REGIME>")

        # Stats
        if stats:
            mom = stats.get('momentum')
            vol = stats.get('volatility')
            parts.append(f"\n<TICKER_STATS>")
            parts.append(f"MOMENTUM_20D: {mom*100:+.2f}%" if mom else "MOMENTUM_20D: N/A")
            parts.append(f"VOLATILITY_20D: {vol*100:.2f}%" if vol else "VOLATILITY_20D: N/A")
            parts.append(f"</TICKER_STATS>")

        # Correlations
        if correlations:
            parts.append(f"\n<CORRELATIONS>")
            for c in correlations[:3]:
                parts.append(f"  {ticker}-{c['ticker']}: r={c['pearson']:.3f}" if c['pearson'] else "")
            parts.append(f"</CORRELATIONS>")

        # News
        parts.append(f"\n<NEWS>")
        parts.append(news[:1000])
        parts.append(f"</NEWS>")

        return "\n".join(parts)

    def get_all_tickers(self) -> List[str]:
        """Get all available tickers."""
        cursor = self.conn.execute("""
            SELECT DISTINCT ticker FROM quant_daily_stats
            ORDER BY ticker
        """)
        return [row[0] for row in cursor.fetchall()]


# =============================================================================
# Portfolio Selector
# =============================================================================

class PortfolioSelector:
    """Select best assets using Promethee + SVM."""

    def __init__(
        self,
        model_path: str,
        svm_path: str,
        device: str = None
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load tokenizer
        logger.info(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Promethee
        base_model = MambaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)

        # Load checkpoint
        checkpoint_path = Path(model_path) / "promethee_final.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            embed_dim = checkpoint.get('embed_dim', 256)
            self.model = PrometheeModel(base_model, embed_dim=embed_dim).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model = PrometheeModel(base_model, embed_dim=256).to(self.device)

        self.model.eval()

        # Load SVM
        logger.info(f"Loading SVM from {svm_path}...")
        svm_data = joblib.load(svm_path)
        self.svm = svm_data['svm']
        self.scaler = svm_data['scaler']

        # Load config if exists
        config_path = Path(svm_path).with_suffix('.json')
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def get_embedding(self, context: str) -> np.ndarray:
        """Extract embedding from context."""
        inputs = self.tokenizer(
            context,
            return_tensors='pt',
            truncation=True,
            max_length=2048,
            padding='max_length'
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'].to(self.device),
                attention_mask=inputs['attention_mask'].to(self.device),
                return_embeddings=True
            )

        return outputs['embeddings'].cpu().numpy().flatten()

    def get_score(self, embedding: np.ndarray) -> float:
        """Get SVM score for embedding."""
        X = self.scaler.transform(embedding.reshape(1, -1))
        proba = self.svm.predict_proba(X)[0]
        # Score = P(winner) - P(loser)
        return proba[1] - proba[0]

    def rank_assets(
        self,
        contexts: Dict[str, str]
    ) -> List[Tuple[str, float, np.ndarray]]:
        """Rank all assets by SVM score."""
        results = []

        for ticker, context in tqdm(contexts.items(), desc="Ranking assets"):
            embedding = self.get_embedding(context)
            score = self.get_score(embedding)
            results.append((ticker, score, embedding))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def select_portfolio(
        self,
        rankings: List[Tuple[str, float, np.ndarray]],
        top_k: int = 10,
        min_score: float = 0.0
    ) -> Dict[str, float]:
        """Select top-K assets with score > min_score."""
        portfolio = {}

        for ticker, score, _ in rankings[:top_k]:
            if score > min_score:
                portfolio[ticker] = score

        # Normalize weights
        if portfolio:
            total = sum(portfolio.values())
            portfolio = {k: v/total for k, v in portfolio.items()}

        return portfolio


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Promethee Inference Pipeline")
    parser.add_argument("--model_path", type=str, default="models/promethee_semantic")
    parser.add_argument("--svm_path", type=str, default="models/promethee_svm.joblib")
    parser.add_argument("--db_path", type=str, default="data/eodhd_sp500.db")
    parser.add_argument("--thresholds_path", type=str, default="data/thresholds.json")
    parser.add_argument("--date", type=str, required=True, help="Analysis date (YYYY-MM-DD)")
    parser.add_argument("--top_k", type=int, default=10, help="Number of assets to select")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers (default: all)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PROMETHEE INFERENCE PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Date: {args.date}")
    logger.info(f"Top-K: {args.top_k}")

    # Initialize
    context_builder = ContextBuilder(args.db_path, args.thresholds_path)
    selector = PortfolioSelector(args.model_path, args.svm_path)

    # Get tickers
    if args.tickers:
        tickers = args.tickers.split(',')
    else:
        tickers = context_builder.get_all_tickers()
        logger.info(f"Found {len(tickers)} tickers")

    # Build contexts
    logger.info("Building contexts...")
    contexts = {}
    for ticker in tqdm(tickers, desc="Building contexts"):
        contexts[ticker] = context_builder.build_context(ticker, args.date)

    # Rank assets
    logger.info("Ranking assets with SVM...")
    rankings = selector.rank_assets(contexts)

    # Display results
    print("\n" + "=" * 60)
    print("ASSET RANKING")
    print("=" * 60)
    print(f"{'Rank':<6} {'Ticker':<8} {'Score':>10} {'Signal':<12}")
    print("-" * 40)

    for i, (ticker, score, _) in enumerate(rankings[:30], 1):
        if score > 0.5:
            signal = "STRONG LONG"
        elif score > 0.2:
            signal = "LONG"
        elif score > -0.2:
            signal = "NEUTRAL"
        elif score > -0.5:
            signal = "SHORT"
        else:
            signal = "STRONG SHORT"

        print(f"{i:<6} {ticker:<8} {score:>+10.4f} {signal:<12}")

    # Select portfolio
    portfolio = selector.select_portfolio(rankings, top_k=args.top_k)

    print("\n" + "=" * 60)
    print("SELECTED PORTFOLIO")
    print("=" * 60)
    print(f"{'Ticker':<8} {'Weight':>10}")
    print("-" * 20)

    for ticker, weight in sorted(portfolio.items(), key=lambda x: -x[1]):
        print(f"{ticker:<8} {weight:>10.1%}")

    print("-" * 20)
    print(f"{'Total':<8} {sum(portfolio.values()):>10.1%}")

    # Save output
    if args.output:
        output_data = {
            'date': args.date,
            'rankings': [(t, float(s)) for t, s, _ in rankings],
            'portfolio': portfolio,
            'config': {
                'top_k': args.top_k,
                'model_path': args.model_path,
                'svm_path': args.svm_path
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Output saved to: {args.output}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
