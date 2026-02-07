"""
Prepare News Training Data for Prométhée

This script extracts news from the database and creates training data
with ground truth market outcomes (returns, volatility, direction).

Two modes:
1. LOCAL: Create training data with market outcomes as labels
2. RUNPOD: Create data for GLM-4 teacher labeling

Usage:
    # Local mode - use market outcomes as labels
    python scripts/prepare_news_training_data.py \
        --db_path eodhd_sp500.db \
        --output_path data/news_training_data.jsonl \
        --mode local \
        --max_samples 100000

    # RunPod mode - prepare for GLM-4 labeling
    python scripts/prepare_news_training_data.py \
        --db_path eodhd_sp500.db \
        --output_path data/news_for_teacher.jsonl \
        --mode runpod \
        --max_samples 50000
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import random

from tqdm import tqdm
from loguru import logger


def load_prices(conn: sqlite3.Connection) -> Dict[str, Dict]:
    """Load all prices indexed by (ticker, date)."""
    logger.info("Loading price data...")

    cursor = conn.execute("""
        SELECT ticker, date, adjusted_close, volume
        FROM historical_prices
        ORDER BY ticker, date
    """)

    prices = defaultdict(dict)
    for ticker, date, close, volume in cursor:
        # Normalize date format (handle both datetime and date strings)
        if date:
            date = str(date)[:10]
        prices[ticker][date] = {'close': close, 'volume': volume}

    logger.info(f"Loaded prices for {len(prices)} tickers")
    return prices


def compute_market_outcome(
    prices: Dict[str, Dict],
    ticker: str,
    date: str,
    horizon: int = 1
) -> Optional[Dict]:
    """
    Compute market outcome for a given (ticker, date).

    Returns:
        - return_1d: 1-day forward return
        - return_5d: 5-day forward return
        - direction: 'up', 'down', 'neutral'
        - volatility: realized volatility (next 5 days)
    """
    if ticker not in prices:
        return None

    ticker_prices = prices[ticker]
    dates = sorted(ticker_prices.keys())

    if date not in ticker_prices:
        # Find closest date
        close_dates = [d for d in dates if d >= date]
        if not close_dates:
            return None
        date = close_dates[0]

    try:
        date_idx = dates.index(date)
    except ValueError:
        return None

    # Need future data
    if date_idx + 5 >= len(dates):
        return None

    current_price = ticker_prices[date]['close']

    # 1-day return
    next_date = dates[date_idx + 1]
    next_price = ticker_prices[next_date]['close']
    return_1d = (next_price / current_price) - 1

    # 5-day return
    future_date = dates[min(date_idx + 5, len(dates) - 1)]
    future_price = ticker_prices[future_date]['close']
    return_5d = (future_price / current_price) - 1

    # Direction
    if return_1d > 0.005:
        direction = 'up'
    elif return_1d < -0.005:
        direction = 'down'
    else:
        direction = 'neutral'

    # Volatility (next 5 days)
    future_prices = [ticker_prices[dates[i]]['close']
                     for i in range(date_idx, min(date_idx + 6, len(dates)))]
    if len(future_prices) >= 2:
        returns = [(future_prices[i+1] / future_prices[i]) - 1
                   for i in range(len(future_prices) - 1)]
        volatility = sum(r**2 for r in returns) ** 0.5
    else:
        volatility = 0.0

    return {
        'return_1d': round(return_1d * 100, 4),  # In percent
        'return_5d': round(return_5d * 100, 4),
        'direction': direction,
        'volatility': round(volatility * 100, 4)
    }


def create_local_training_sample(
    news: Dict,
    outcome: Dict
) -> Dict:
    """Create a training sample with market outcome as label."""

    # Determine sentiment from outcome
    if outcome['return_1d'] > 1.0:
        sentiment = 'strongly_bullish'
    elif outcome['return_1d'] > 0.2:
        sentiment = 'bullish'
    elif outcome['return_1d'] < -1.0:
        sentiment = 'strongly_bearish'
    elif outcome['return_1d'] < -0.2:
        sentiment = 'bearish'
    else:
        sentiment = 'neutral'

    # Determine impact magnitude
    abs_return = abs(outcome['return_1d'])
    if abs_return > 3.0:
        impact = 'very_high'
    elif abs_return > 1.5:
        impact = 'high'
    elif abs_return > 0.5:
        impact = 'medium'
    else:
        impact = 'low'

    # Create structured label (what Teacher would generate)
    teacher_output = {
        'sentiment': sentiment,
        'impact_magnitude': impact,
        'direction': outcome['direction'],
        'expected_return': f"{outcome['return_1d']:+.2f}%",
        'volatility_forecast': f"{outcome['volatility']:.2f}%",
        'causal_factors': [
            f"News event on {news['date'][:10]}",
            f"Ticker: {news['ticker']}",
            f"Market moved {outcome['direction']} by {abs(outcome['return_1d']):.2f}%"
        ],
        'confidence': 'high' if abs_return > 1.0 else 'medium'
    }

    return {
        'input_text': news['title'] + (f". {news['content'][:2000]}" if news.get('content') else ""),
        'ticker': news['ticker'],
        'date': news['date'][:10],
        'teacher_output': json.dumps(teacher_output),
        'ground_truth': outcome
    }


def create_runpod_sample(news: Dict) -> Dict:
    """Create a sample for GLM-4 teacher labeling."""
    return {
        'id': f"{news['date'][:10]}_{news['ticker']}_{news['id']}",
        'input_text': news['title'] + (f". {news['content'][:3000]}" if news.get('content') else ""),
        'ticker': news['ticker'],
        'date': news['date'][:10]
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare news training data for Prométhée")
    parser.add_argument("--db_path", type=str, required=True, help="Path to SQLite database")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--mode", type=str, choices=['local', 'runpod'], default='local',
                        help="Mode: 'local' uses market outcomes, 'runpod' for GLM-4 labeling")
    parser.add_argument("--max_samples", type=int, default=100000, help="Max samples to generate")
    parser.add_argument("--start_date", type=str, default="2010-01-01", help="Start date")
    parser.add_argument("--end_date", type=str, default="2025-12-31", help="End date")
    parser.add_argument("--min_title_length", type=int, default=20, help="Min title length")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Random sampling rate")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🔥 Prométhée - News Training Data Preparation")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Max samples: {args.max_samples:,}")
    logger.info("")

    # Connect to database
    conn = sqlite3.connect(args.db_path)

    # Load prices for local mode
    prices = None
    if args.mode == 'local':
        prices = load_prices(conn)

    # Query news
    logger.info("Loading news articles...")
    cursor = conn.execute("""
        SELECT id, date, ticker, title, content
        FROM news
        WHERE date >= ? AND date <= ?
        ORDER BY date
    """, (args.start_date, args.end_date))

    # Process news
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples_written = 0
    skipped_no_outcome = 0
    skipped_short = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for row in tqdm(cursor, desc="Processing news"):
            if samples_written >= args.max_samples:
                break

            # Random sampling
            if args.sample_rate < 1.0 and random.random() > args.sample_rate:
                continue

            news = {
                'id': row[0],
                'date': row[1],
                'ticker': row[2],
                'title': row[3] or "",
                'content': row[4] or ""
            }

            # Filter short titles
            if len(news['title']) < args.min_title_length:
                skipped_short += 1
                continue

            if args.mode == 'local':
                # Get market outcome
                date = news['date'][:10] if news['date'] else None
                if not date:
                    continue

                outcome = compute_market_outcome(prices, news['ticker'], date)
                if outcome is None:
                    skipped_no_outcome += 1
                    continue

                sample = create_local_training_sample(news, outcome)
            else:
                sample = create_runpod_sample(news)

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            samples_written += 1

            if samples_written % 10000 == 0:
                logger.info(f"Written {samples_written:,} samples...")

    conn.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  Samples written: {samples_written:,}")
    logger.info(f"  Skipped (no outcome): {skipped_no_outcome:,}")
    logger.info(f"  Skipped (short title): {skipped_short:,}")
    logger.info(f"  Output: {output_path}")
    logger.info("=" * 60)

    # Show sample
    logger.info("\nSample output:")
    with open(output_path, 'r') as f:
        sample = json.loads(f.readline())
        logger.info(json.dumps(sample, indent=2, ensure_ascii=False)[:500] + "...")


if __name__ == "__main__":
    main()
