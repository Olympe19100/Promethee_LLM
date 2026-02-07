"""
Prepare Unified Training Data for Prométhée

Combines all 3 data sources:
1. SEC 10-K filings (fundamental analysis)
2. News articles (event-driven analysis)
3. Quantitative context (market regime, stats, correlations)

Each training sample includes:
- Text input (news or 10-K excerpt)
- Quantitative context (regime, volatility, sector performance)
- Ground truth market outcome (returns, direction, volatility)

Usage:
    python scripts/prepare_unified_training_data.py \
        --db_path eodhd_sp500.db \
        --output_path data/promethee_unified_training.jsonl \
        --max_samples 300000
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import random
import re

from tqdm import tqdm
from loguru import logger


# =============================================================================
# Data Loaders
# =============================================================================

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
        if date:
            date = str(date)[:10]
        prices[ticker][date] = {'close': close, 'volume': volume}

    logger.info(f"Loaded prices for {len(prices)} tickers")
    return prices


def load_quant_daily_stats(conn: sqlite3.Connection) -> Dict[Tuple[str, str], Dict]:
    """Load quantitative daily statistics."""
    logger.info("Loading quantitative daily stats...")

    cursor = conn.execute("""
        SELECT date, ticker, mean_20d, std_20d, skew_20d, kurt_20d, var_5pct, momentum_20d
        FROM quant_daily_stats
    """)

    stats = {}
    for row in cursor:
        date, ticker = str(row[0])[:10], row[1]
        stats[(ticker, date)] = {
            'mean_20d': row[2],
            'std_20d': row[3],
            'skew_20d': row[4],
            'kurt_20d': row[5],
            'var_5pct': row[6],
            'momentum_20d': row[7]
        }

    logger.info(f"Loaded {len(stats):,} daily stat records")
    return stats


def load_quant_regime(conn: sqlite3.Connection) -> Dict[str, Dict]:
    """Load market regime data."""
    logger.info("Loading market regime data...")

    cursor = conn.execute("""
        SELECT date, regime, regime_duration, avg_correlation, dispersion,
               vix_level, yield_spread, hmm_state
        FROM quant_regime
    """)

    regimes = {}
    for row in cursor:
        date = str(row[0])[:10]
        regimes[date] = {
            'regime': row[1],
            'regime_duration': row[2],
            'avg_correlation': row[3],
            'dispersion': row[4],
            'vix_level': row[5],
            'yield_spread': row[6],
            'hmm_state': row[7]
        }

    logger.info(f"Loaded {len(regimes):,} regime records")
    return regimes


def load_quant_sector(conn: sqlite3.Connection) -> Dict[Tuple[str, str], Dict]:
    """Load sector performance data."""
    logger.info("Loading sector data...")

    cursor = conn.execute("""
        SELECT date, sector, return_1d, return_20d, volatility, beta, internal_corr
        FROM quant_sector
    """)

    sectors = {}
    for row in cursor:
        date, sector = str(row[0])[:10], row[1]
        sectors[(sector, date)] = {
            'sector_return_1d': row[2],
            'sector_return_20d': row[3],
            'sector_volatility': row[4],
            'sector_beta': row[5],
            'sector_internal_corr': row[6]
        }

    logger.info(f"Loaded {len(sectors):,} sector records")
    return sectors


def load_ticker_sectors(conn: sqlite3.Connection) -> Dict[str, str]:
    """Load ticker to sector mapping."""
    cursor = conn.execute("""
        SELECT ticker, sector FROM fundamentals_general
    """)

    mapping = {}
    for ticker, sector in cursor:
        if sector:
            mapping[ticker] = sector

    logger.info(f"Loaded sector mapping for {len(mapping)} tickers")
    return mapping


# =============================================================================
# Market Outcome Computation
# =============================================================================

def compute_market_outcome(
    prices: Dict[str, Dict],
    ticker: str,
    date: str
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
        close_dates = [d for d in dates if d >= date]
        if not close_dates:
            return None
        date = close_dates[0]

    try:
        date_idx = dates.index(date)
    except ValueError:
        return None

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
        'return_1d': round(return_1d * 100, 4),
        'return_5d': round(return_5d * 100, 4),
        'direction': direction,
        'volatility': round(volatility * 100, 4)
    }


# =============================================================================
# Quantitative Context Builder
# =============================================================================

def build_quant_context(
    ticker: str,
    date: str,
    daily_stats: Dict,
    regimes: Dict,
    sectors: Dict,
    ticker_sectors: Dict
) -> Dict:
    """Build quantitative context for a sample."""

    context = {
        'has_quant_context': False,
        'regime': None,
        'vix': None,
        'market_correlation': None,
        'ticker_momentum': None,
        'ticker_volatility': None,
        'sector': None,
        'sector_performance': None
    }

    # Market regime
    if date in regimes:
        r = regimes[date]
        context['regime'] = r['regime']
        context['vix'] = r['vix_level']
        context['market_correlation'] = r['avg_correlation']
        context['yield_spread'] = r['yield_spread']
        context['has_quant_context'] = True

    # Ticker daily stats
    if (ticker, date) in daily_stats:
        s = daily_stats[(ticker, date)]
        context['ticker_momentum'] = s['momentum_20d']
        context['ticker_volatility'] = s['std_20d']
        context['ticker_skew'] = s['skew_20d']
        context['ticker_var'] = s['var_5pct']
        context['has_quant_context'] = True

    # Sector performance
    sector = ticker_sectors.get(ticker)
    if sector:
        context['sector'] = sector
        if (sector, date) in sectors:
            sec = sectors[(sector, date)]
            context['sector_performance'] = sec['sector_return_20d']
            context['sector_volatility'] = sec['sector_volatility']
            context['has_quant_context'] = True

    return context


def format_quant_context_text(context: Dict) -> str:
    """Format quantitative context as natural language."""
    if not context['has_quant_context']:
        return ""

    parts = []

    if context['regime']:
        parts.append(f"Market regime: {context['regime']}")

    if context['vix'] is not None:
        vix = context['vix']
        if vix < 15:
            vix_desc = "low volatility"
        elif vix < 25:
            vix_desc = "moderate volatility"
        else:
            vix_desc = "high volatility"
        parts.append(f"VIX: {vix:.1f} ({vix_desc})")

    if context['market_correlation'] is not None:
        corr = context['market_correlation']
        if corr > 0.5:
            corr_desc = "high correlation (risk-off)"
        elif corr > 0.3:
            corr_desc = "moderate correlation"
        else:
            corr_desc = "low correlation (stock-picking environment)"
        parts.append(f"Market correlation: {corr:.2f} ({corr_desc})")

    if context['ticker_momentum'] is not None:
        mom = context['ticker_momentum']
        if mom > 0.02:
            mom_desc = "strong upward momentum"
        elif mom > 0:
            mom_desc = "positive momentum"
        elif mom > -0.02:
            mom_desc = "negative momentum"
        else:
            mom_desc = "strong downward momentum"
        parts.append(f"Stock momentum: {mom*100:.1f}% ({mom_desc})")

    if context['sector'] and context['sector_performance'] is not None:
        perf = context['sector_performance']
        parts.append(f"Sector ({context['sector']}): {perf*100:.1f}% over 20 days")

    return " | ".join(parts)


# =============================================================================
# Training Sample Creation
# =============================================================================

def determine_sentiment_and_impact(outcome: Dict) -> Tuple[str, str]:
    """Determine sentiment and impact from market outcome."""
    ret = outcome['return_1d']

    if ret > 1.0:
        sentiment = 'strongly_bullish'
    elif ret > 0.2:
        sentiment = 'bullish'
    elif ret < -1.0:
        sentiment = 'strongly_bearish'
    elif ret < -0.2:
        sentiment = 'bearish'
    else:
        sentiment = 'neutral'

    abs_ret = abs(ret)
    if abs_ret > 3.0:
        impact = 'very_high'
    elif abs_ret > 1.5:
        impact = 'high'
    elif abs_ret > 0.5:
        impact = 'medium'
    else:
        impact = 'low'

    return sentiment, impact


def create_training_sample(
    source_type: str,
    text: str,
    ticker: str,
    date: str,
    outcome: Dict,
    quant_context: Dict
) -> Dict:
    """Create a unified training sample."""

    sentiment, impact = determine_sentiment_and_impact(outcome)

    # Build structured teacher output
    teacher_output = {
        'sentiment': sentiment,
        'impact_magnitude': impact,
        'direction': outcome['direction'],
        'expected_return': f"{outcome['return_1d']:+.2f}%",
        'volatility_forecast': f"{outcome['volatility']:.2f}%",
        'confidence': 'high' if abs(outcome['return_1d']) > 1.0 else 'medium'
    }

    # Add causal factors based on context
    causal_factors = []
    if quant_context['regime']:
        causal_factors.append(f"Market in {quant_context['regime']} regime")
    if quant_context['ticker_momentum'] is not None:
        mom = quant_context['ticker_momentum']
        if abs(mom) > 0.01:
            causal_factors.append(f"Stock {'positive' if mom > 0 else 'negative'} momentum ({mom*100:.1f}%)")
    if quant_context['sector']:
        causal_factors.append(f"Sector: {quant_context['sector']}")

    teacher_output['causal_factors'] = causal_factors if causal_factors else ["Event-driven move"]

    # Format quantitative context
    quant_text = format_quant_context_text(quant_context)

    return {
        'source_type': source_type,  # 'news' or 'sec_10k'
        'input_text': text[:8000],
        'quant_context': quant_text,
        'ticker': ticker,
        'date': date,
        'teacher_output': json.dumps(teacher_output),
        'ground_truth': outcome,
        'quant_features': {
            'regime': quant_context.get('regime'),
            'vix': quant_context.get('vix'),
            'momentum': quant_context.get('ticker_momentum'),
            'sector': quant_context.get('sector')
        }
    }


# =============================================================================
# Main Processing
# =============================================================================

def process_news(
    conn: sqlite3.Connection,
    prices: Dict,
    daily_stats: Dict,
    regimes: Dict,
    sectors: Dict,
    ticker_sectors: Dict,
    start_date: str,
    end_date: str,
    max_samples: int,
    sample_rate: float = 1.0
) -> List[Dict]:
    """Process news articles into training samples."""

    logger.info("Processing news articles...")

    cursor = conn.execute("""
        SELECT id, date, ticker, title, content
        FROM news
        WHERE date >= ? AND date <= ?
        ORDER BY date
    """, (start_date, end_date))

    samples = []
    skipped = 0

    for row in tqdm(cursor, desc="News"):
        if len(samples) >= max_samples:
            break

        if sample_rate < 1.0 and random.random() > sample_rate:
            continue

        news_id, date, ticker, title, content = row

        if not title or len(title) < 20:
            skipped += 1
            continue

        date = str(date)[:10] if date else None
        if not date:
            continue

        # Get market outcome
        outcome = compute_market_outcome(prices, ticker, date)
        if outcome is None:
            skipped += 1
            continue

        # Build quantitative context
        quant_context = build_quant_context(
            ticker, date, daily_stats, regimes, sectors, ticker_sectors
        )

        # Create text
        text = title
        if content:
            text += f". {content[:3000]}"

        sample = create_training_sample(
            source_type='news',
            text=text,
            ticker=ticker,
            date=date,
            outcome=outcome,
            quant_context=quant_context
        )
        samples.append(sample)

    logger.info(f"Processed {len(samples):,} news samples (skipped {skipped:,})")
    return samples


def process_sec_filings(
    conn: sqlite3.Connection,
    prices: Dict,
    daily_stats: Dict,
    regimes: Dict,
    sectors: Dict,
    ticker_sectors: Dict,
    max_samples: int
) -> List[Dict]:
    """Process SEC 10-K filings into training samples."""

    logger.info("Processing SEC 10-K filings...")

    cursor = conn.execute("""
        SELECT ticker, filing_date, content
        FROM sec_filings
        WHERE filing_type = '10-K' AND content IS NOT NULL
    """)

    samples = []
    skipped = 0

    for row in tqdm(cursor, desc="SEC 10-K"):
        if len(samples) >= max_samples:
            break

        ticker, filing_date, content = row

        if not content or len(content) < 1000:
            skipped += 1
            continue

        # Try to find date from content if filing_date is null
        date = None
        if filing_date:
            date = str(filing_date)[:10]
        else:
            # Extract date from content header
            match = re.search(r'FILED AS OF DATE:\s*(\d{8})', content)
            if match:
                d = match.group(1)
                date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

        if not date:
            skipped += 1
            continue

        # Get market outcome
        outcome = compute_market_outcome(prices, ticker, date)
        if outcome is None:
            skipped += 1
            continue

        # Build quantitative context
        quant_context = build_quant_context(
            ticker, date, daily_stats, regimes, sectors, ticker_sectors
        )

        # Extract meaningful sections from 10-K
        # Focus on MD&A (Management Discussion), Risk Factors, Business Description
        text_parts = []

        # Try to extract key sections
        content_lower = content.lower()

        # Business description
        if 'item 1.' in content_lower:
            start = content_lower.find('item 1.')
            end = content_lower.find('item 2.', start) if start > 0 else -1
            if start > 0 and end > start:
                text_parts.append(content[start:min(end, start+5000)])

        # Risk factors
        if 'risk factors' in content_lower:
            start = content_lower.find('risk factors')
            text_parts.append(content[start:start+3000])

        # If no sections found, use beginning of document
        if not text_parts:
            text_parts.append(content[:6000])

        text = f"SEC 10-K Filing for {ticker}. " + " ".join(text_parts)

        sample = create_training_sample(
            source_type='sec_10k',
            text=text,
            ticker=ticker,
            date=date,
            outcome=outcome,
            quant_context=quant_context
        )
        samples.append(sample)

    logger.info(f"Processed {len(samples):,} SEC samples (skipped {skipped:,})")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare unified training data for Prométhée")
    parser.add_argument("--db_path", type=str, required=True, help="Path to SQLite database")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--max_news", type=int, default=250000, help="Max news samples")
    parser.add_argument("--max_sec", type=int, default=50000, help="Max SEC samples")
    parser.add_argument("--start_date", type=str, default="2010-01-01", help="Start date")
    parser.add_argument("--end_date", type=str, default="2025-12-31", help="End date")
    parser.add_argument("--news_sample_rate", type=float, default=0.2, help="News sampling rate")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PROMETHEE - Unified Training Data Preparation")
    logger.info("=" * 70)
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Max news: {args.max_news:,}, Max SEC: {args.max_sec:,}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info("")

    # Connect to database
    conn = sqlite3.connect(args.db_path)

    # Load all reference data
    prices = load_prices(conn)
    daily_stats = load_quant_daily_stats(conn)
    regimes = load_quant_regime(conn)
    sectors = load_quant_sector(conn)
    ticker_sectors = load_ticker_sectors(conn)

    logger.info("")

    # Process news
    news_samples = process_news(
        conn, prices, daily_stats, regimes, sectors, ticker_sectors,
        args.start_date, args.end_date, args.max_news, args.news_sample_rate
    )

    # Process SEC filings
    sec_samples = process_sec_filings(
        conn, prices, daily_stats, regimes, sectors, ticker_sectors,
        args.max_sec
    )

    conn.close()

    # Combine samples
    all_samples = news_samples + sec_samples

    if args.shuffle:
        random.shuffle(all_samples)

    # Write output
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Statistics
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total samples: {len(all_samples):,}")
    logger.info(f"  - News: {len(news_samples):,}")
    logger.info(f"  - SEC 10-K: {len(sec_samples):,}")

    # Sentiment distribution
    sentiments = defaultdict(int)
    for s in all_samples:
        teacher = json.loads(s['teacher_output'])
        sentiments[teacher['sentiment']] += 1

    logger.info("")
    logger.info("Sentiment distribution:")
    for sent, count in sorted(sentiments.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_samples)
        logger.info(f"  {sent}: {count:,} ({pct:.1f}%)")

    # Quant context coverage
    with_quant = sum(1 for s in all_samples if s['quant_features']['regime'] is not None)
    logger.info("")
    logger.info(f"Samples with quant context: {with_quant:,} ({100*with_quant/len(all_samples):.1f}%)")

    logger.info("")
    logger.info(f"Output saved to: {output_path}")

    # Show sample
    logger.info("")
    logger.info("Sample output:")
    sample = all_samples[0]
    logger.info(f"  Source: {sample['source_type']}")
    logger.info(f"  Ticker: {sample['ticker']}, Date: {sample['date']}")
    logger.info(f"  Quant: {sample['quant_context'][:100]}...")
    logger.info(f"  Text: {sample['input_text'][:150]}...")
    teacher = json.loads(sample['teacher_output'])
    logger.info(f"  Teacher: sentiment={teacher['sentiment']}, impact={teacher['impact_magnitude']}")


if __name__ == "__main__":
    main()
