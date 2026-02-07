"""
Prepare SOTA Training Data for Promethee - DATA-DRIVEN VERSION

ALL parameters are computed from data distributions:
- Thresholds based on percentiles (not hardcoded)
- Z-scores relative to historical norms
- Dynamic regime classification
- Adaptive position sizing based on volatility

Usage:
    python scripts/prepare_promethee_training.py \
        --db_path eodhd_sp500.db \
        --output_path data/promethee_semantic_training.jsonl \
        --max_samples 300000
"""

import argparse
import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import random
import re

from tqdm import tqdm
from loguru import logger


# =============================================================================
# Data-Driven Thresholds Calculator
# =============================================================================

class DataDrivenThresholds:
    """
    Computes all thresholds from actual data distributions.
    No hardcoded values - everything is percentile-based.
    """

    def __init__(self):
        self.thresholds = {}
        self.distributions = {}

    def compute_from_data(self, conn: sqlite3.Connection):
        """Compute all thresholds from database."""
        logger.info("Computing data-driven thresholds...")

        # 1. VIX thresholds
        self._compute_vix_thresholds(conn)

        # 2. Correlation thresholds
        self._compute_correlation_thresholds(conn)

        # 3. Momentum thresholds
        self._compute_momentum_thresholds(conn)

        # 4. Volatility thresholds
        self._compute_volatility_thresholds(conn)

        # 5. Skewness/Kurtosis thresholds
        self._compute_distribution_thresholds(conn)

        # 6. Z-score thresholds for cointegration
        self._compute_zscore_thresholds(conn)

        # 7. Tail dependency thresholds
        self._compute_tail_dep_thresholds(conn)

        # 8. Return thresholds
        self._compute_return_thresholds(conn)

        logger.info("Data-driven thresholds computed:")
        for key, value in self.thresholds.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        logger.info(f"    {k}: {v:.4f}")
                    else:
                        logger.info(f"    {k}: {v}")
            else:
                logger.info(f"  {key}: {value}")

    def _compute_vix_thresholds(self, conn):
        """VIX percentiles for regime classification."""
        cursor = conn.execute("SELECT vix_level FROM quant_regime WHERE vix_level IS NOT NULL")
        values = [r[0] for r in cursor.fetchall()]

        if values:
            self.distributions['vix'] = values
            self.thresholds['vix'] = {
                'p25': float(np.percentile(values, 25)),  # Low VIX
                'p50': float(np.percentile(values, 50)),  # Normal
                'p75': float(np.percentile(values, 75)),  # Elevated
                'p90': float(np.percentile(values, 90)),  # High
                'p95': float(np.percentile(values, 95)),  # Extreme
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        else:
            # Fallback if no data
            self.thresholds['vix'] = {'p25': 13, 'p50': 17, 'p75': 22, 'p90': 28, 'p95': 35, 'mean': 18, 'std': 6}

    def _compute_correlation_thresholds(self, conn):
        """Market correlation percentiles."""
        cursor = conn.execute("SELECT avg_correlation FROM quant_regime WHERE avg_correlation IS NOT NULL")
        values = [r[0] for r in cursor.fetchall()]

        if values:
            self.distributions['correlation'] = values
            self.thresholds['correlation'] = {
                'p25': float(np.percentile(values, 25)),  # Low (stock-picking)
                'p50': float(np.percentile(values, 50)),  # Normal
                'p75': float(np.percentile(values, 75)),  # High (risk-off)
                'p90': float(np.percentile(values, 90)),  # Very high
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        else:
            self.thresholds['correlation'] = {'p25': 0.2, 'p50': 0.35, 'p75': 0.5, 'p90': 0.65, 'mean': 0.35, 'std': 0.15}

    def _compute_momentum_thresholds(self, conn):
        """Momentum percentiles by ticker."""
        cursor = conn.execute("SELECT momentum_20d FROM quant_daily_stats WHERE momentum_20d IS NOT NULL")
        values = [r[0] for r in cursor.fetchall()]

        if values:
            self.distributions['momentum'] = values
            self.thresholds['momentum'] = {
                'p5': float(np.percentile(values, 5)),    # Strong down
                'p10': float(np.percentile(values, 10)),  # Down
                'p25': float(np.percentile(values, 25)),  # Weak down
                'p75': float(np.percentile(values, 75)),  # Weak up
                'p90': float(np.percentile(values, 90)),  # Up
                'p95': float(np.percentile(values, 95)),  # Strong up
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        else:
            self.thresholds['momentum'] = {'p5': -0.15, 'p10': -0.08, 'p25': -0.02, 'p75': 0.02, 'p90': 0.08, 'p95': 0.15, 'mean': 0, 'std': 0.08}

    def _compute_volatility_thresholds(self, conn):
        """Volatility percentiles."""
        cursor = conn.execute("SELECT std_20d FROM quant_daily_stats WHERE std_20d IS NOT NULL")
        values = [r[0] for r in cursor.fetchall()]

        if values:
            self.distributions['volatility'] = values
            self.thresholds['volatility'] = {
                'p25': float(np.percentile(values, 25)),  # Low vol
                'p50': float(np.percentile(values, 50)),  # Normal
                'p75': float(np.percentile(values, 75)),  # High
                'p90': float(np.percentile(values, 90)),  # Very high
                'p95': float(np.percentile(values, 95)),  # Extreme
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
        else:
            self.thresholds['volatility'] = {'p25': 0.015, 'p50': 0.02, 'p75': 0.03, 'p90': 0.04, 'p95': 0.05, 'mean': 0.025, 'std': 0.012}

    def _compute_distribution_thresholds(self, conn):
        """Skewness and Kurtosis thresholds."""
        # Skewness
        cursor = conn.execute("SELECT skew_20d FROM quant_daily_stats WHERE skew_20d IS NOT NULL")
        skew_values = [r[0] for r in cursor.fetchall()]

        if skew_values:
            self.distributions['skewness'] = skew_values
            self.thresholds['skewness'] = {
                'p5': float(np.percentile(skew_values, 5)),    # Strong negative
                'p10': float(np.percentile(skew_values, 10)),  # Negative
                'p90': float(np.percentile(skew_values, 90)),  # Positive
                'p95': float(np.percentile(skew_values, 95)),  # Strong positive
                'mean': float(np.mean(skew_values)),
                'std': float(np.std(skew_values))
            }
        else:
            self.thresholds['skewness'] = {'p5': -1.5, 'p10': -0.8, 'p90': 0.8, 'p95': 1.5, 'mean': 0, 'std': 0.8}

        # Kurtosis
        cursor = conn.execute("SELECT kurt_20d FROM quant_daily_stats WHERE kurt_20d IS NOT NULL")
        kurt_values = [r[0] for r in cursor.fetchall()]

        if kurt_values:
            self.distributions['kurtosis'] = kurt_values
            self.thresholds['kurtosis'] = {
                'p50': float(np.percentile(kurt_values, 50)),  # Normal
                'p75': float(np.percentile(kurt_values, 75)),  # High
                'p90': float(np.percentile(kurt_values, 90)),  # Very high
                'p95': float(np.percentile(kurt_values, 95)),  # Extreme
                'mean': float(np.mean(kurt_values)),
                'std': float(np.std(kurt_values))
            }
        else:
            self.thresholds['kurtosis'] = {'p50': 3, 'p75': 4, 'p90': 5, 'p95': 7, 'mean': 3.5, 'std': 2}

    def _compute_zscore_thresholds(self, conn):
        """Z-score thresholds for cointegration signals."""
        cursor = conn.execute("""
            SELECT spread_zscore FROM quant_pairwise
            WHERE is_cointegrated = 1 AND spread_zscore IS NOT NULL
        """)
        values = [r[0] for r in cursor.fetchall()]

        if values:
            self.distributions['zscore'] = values
            # For z-scores, we use symmetric percentiles
            abs_values = [abs(v) for v in values]
            self.thresholds['zscore'] = {
                'p75': float(np.percentile(abs_values, 75)),  # Moderate signal
                'p90': float(np.percentile(abs_values, 90)),  # Strong signal
                'p95': float(np.percentile(abs_values, 95)),  # Very strong
                'mean_abs': float(np.mean(abs_values)),
                'std': float(np.std(values))
            }
        else:
            self.thresholds['zscore'] = {'p75': 1.5, 'p90': 2.0, 'p95': 2.5, 'mean_abs': 1.0, 'std': 1.2}

    def _compute_tail_dep_thresholds(self, conn):
        """Tail dependency thresholds."""
        cursor = conn.execute("""
            SELECT tail_dep_lower, tail_dep_upper FROM quant_pairwise
            WHERE tail_dep_lower IS NOT NULL OR tail_dep_upper IS NOT NULL
        """)
        lower_vals = []
        upper_vals = []
        for row in cursor.fetchall():
            if row[0] is not None:
                lower_vals.append(row[0])
            if row[1] is not None:
                upper_vals.append(row[1])

        if lower_vals:
            self.distributions['tail_dep_lower'] = lower_vals
            self.thresholds['tail_dep'] = {
                'lower_p75': float(np.percentile(lower_vals, 75)),
                'lower_p90': float(np.percentile(lower_vals, 90)),
                'upper_p75': float(np.percentile(upper_vals, 75)) if upper_vals else 0.5,
                'upper_p90': float(np.percentile(upper_vals, 90)) if upper_vals else 0.7,
                'lower_mean': float(np.mean(lower_vals)),
                'upper_mean': float(np.mean(upper_vals)) if upper_vals else 0.3
            }
        else:
            self.thresholds['tail_dep'] = {'lower_p75': 0.4, 'lower_p90': 0.6, 'upper_p75': 0.4, 'upper_p90': 0.6, 'lower_mean': 0.3, 'upper_mean': 0.3}

    def _compute_return_thresholds(self, conn):
        """Return thresholds for direction classification."""
        cursor = conn.execute("""
            SELECT h2.adjusted_close / h1.adjusted_close - 1 as ret_1d
            FROM historical_prices h1
            JOIN historical_prices h2 ON h1.ticker = h2.ticker
                AND date(h2.date) = date(h1.date, '+1 day')
            WHERE h1.adjusted_close > 0 AND h2.adjusted_close > 0
            LIMIT 1000000
        """)
        values = [r[0] for r in cursor.fetchall() if r[0] is not None and abs(r[0]) < 0.5]

        if values:
            self.distributions['returns'] = values
            self.thresholds['returns'] = {
                'p5': float(np.percentile(values, 5)),
                'p10': float(np.percentile(values, 10)),
                'p25': float(np.percentile(values, 25)),
                'p75': float(np.percentile(values, 75)),
                'p90': float(np.percentile(values, 90)),
                'p95': float(np.percentile(values, 95)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                # Neutral zone = within 0.5 std of mean
                'neutral_low': float(np.mean(values) - 0.5 * np.std(values)),
                'neutral_high': float(np.mean(values) + 0.5 * np.std(values))
            }
        else:
            self.thresholds['returns'] = {
                'p5': -0.03, 'p10': -0.02, 'p25': -0.008, 'p75': 0.008, 'p90': 0.02, 'p95': 0.03,
                'mean': 0.0005, 'std': 0.015, 'neutral_low': -0.005, 'neutral_high': 0.005
            }

    # =========================================================================
    # Data-Driven Classification Methods
    # =========================================================================

    def classify_vix(self, vix: float) -> Tuple[str, float]:
        """Classify VIX level using data-driven percentiles."""
        if vix is None:
            return "UNKNOWN", 0.5

        t = self.thresholds['vix']
        zscore = (vix - t['mean']) / t['std'] if t['std'] > 0 else 0

        if vix <= t['p25']:
            return "LOW", zscore
        elif vix <= t['p50']:
            return "NORMAL_LOW", zscore
        elif vix <= t['p75']:
            return "NORMAL", zscore
        elif vix <= t['p90']:
            return "ELEVATED", zscore
        elif vix <= t['p95']:
            return "HIGH", zscore
        else:
            return "EXTREME", zscore

    def classify_correlation(self, corr: float) -> Tuple[str, float]:
        """Classify market correlation level."""
        if corr is None:
            return "UNKNOWN", 0.5

        t = self.thresholds['correlation']
        zscore = (corr - t['mean']) / t['std'] if t['std'] > 0 else 0

        if corr <= t['p25']:
            return "LOW", zscore  # Stock-picking environment
        elif corr <= t['p50']:
            return "NORMAL_LOW", zscore
        elif corr <= t['p75']:
            return "NORMAL_HIGH", zscore
        else:
            return "HIGH", zscore  # Risk-off, macro-driven

    def classify_momentum(self, momentum: float) -> Tuple[str, float]:
        """Classify momentum using percentiles."""
        if momentum is None:
            return "NEUTRAL", 0.0

        t = self.thresholds['momentum']
        zscore = (momentum - t['mean']) / t['std'] if t['std'] > 0 else 0

        if momentum <= t['p5']:
            return "STRONG_DOWN", zscore
        elif momentum <= t['p10']:
            return "DOWN", zscore
        elif momentum <= t['p25']:
            return "WEAK_DOWN", zscore
        elif momentum <= t['p75']:
            return "NEUTRAL", zscore
        elif momentum <= t['p90']:
            return "WEAK_UP", zscore
        elif momentum <= t['p95']:
            return "UP", zscore
        else:
            return "STRONG_UP", zscore

    def classify_volatility(self, vol: float) -> Tuple[str, float]:
        """Classify volatility level."""
        if vol is None:
            return "UNKNOWN", 0.5

        t = self.thresholds['volatility']
        zscore = (vol - t['mean']) / t['std'] if t['std'] > 0 else 0

        if vol <= t['p25']:
            return "LOW", zscore
        elif vol <= t['p50']:
            return "NORMAL", zscore
        elif vol <= t['p75']:
            return "ELEVATED", zscore
        elif vol <= t['p90']:
            return "HIGH", zscore
        else:
            return "EXTREME", zscore

    def classify_skewness(self, skew: float) -> Tuple[str, float]:
        """Classify skewness."""
        if skew is None:
            return "NORMAL", 0.0

        t = self.thresholds['skewness']
        zscore = (skew - t['mean']) / t['std'] if t['std'] > 0 else 0

        if skew <= t['p5']:
            return "STRONG_LEFT_TAIL", zscore
        elif skew <= t['p10']:
            return "LEFT_TAIL", zscore
        elif skew >= t['p95']:
            return "STRONG_RIGHT_SKEW", zscore
        elif skew >= t['p90']:
            return "RIGHT_SKEW", zscore
        else:
            return "NORMAL", zscore

    def classify_kurtosis(self, kurt: float) -> Tuple[str, float]:
        """Classify kurtosis (tail fatness)."""
        if kurt is None:
            return "NORMAL", 0.0

        t = self.thresholds['kurtosis']
        zscore = (kurt - t['mean']) / t['std'] if t['std'] > 0 else 0

        if kurt <= t['p50']:
            return "NORMAL", zscore
        elif kurt <= t['p75']:
            return "ELEVATED", zscore
        elif kurt <= t['p90']:
            return "FAT_TAILS", zscore
        else:
            return "EXTREME_TAILS", zscore

    def classify_zscore(self, zscore_val: float) -> Tuple[str, float]:
        """Classify cointegration z-score signal strength."""
        if zscore_val is None:
            return "NONE", 0.0

        t = self.thresholds['zscore']
        abs_z = abs(zscore_val)
        strength = abs_z / t['std'] if t['std'] > 0 else abs_z

        if abs_z < t['p75']:
            return "WEAK", strength
        elif abs_z < t['p90']:
            return "MODERATE", strength
        elif abs_z < t['p95']:
            return "STRONG", strength
        else:
            return "EXTREME", strength

    def classify_tail_dep(self, tail_val: float, dep_type: str = 'lower') -> Tuple[str, float]:
        """Classify tail dependency level."""
        if tail_val is None:
            return "NONE", 0.0

        t = self.thresholds['tail_dep']
        key_p75 = f'{dep_type}_p75'
        key_p90 = f'{dep_type}_p90'
        key_mean = f'{dep_type}_mean'

        if tail_val < t.get(key_p75, 0.4):
            return "LOW", tail_val
        elif tail_val < t.get(key_p90, 0.6):
            return "MODERATE", tail_val
        else:
            return "HIGH", tail_val

    def classify_return(self, ret: float) -> str:
        """Classify return direction."""
        if ret is None:
            return "neutral"

        t = self.thresholds['returns']

        if ret < t['neutral_low']:
            return "down"
        elif ret > t['neutral_high']:
            return "up"
        else:
            return "neutral"

    def get_regime_mode(self, vix: float, corr: float) -> Tuple[str, float]:
        """Determine market regime mode from VIX and correlation."""
        vix_class, vix_z = self.classify_vix(vix)
        corr_class, corr_z = self.classify_correlation(corr)

        # Combined score (higher = more defensive)
        risk_score = (abs(vix_z) if vix_z > 0 else 0) + (abs(corr_z) if corr_z > 0 else 0)

        if vix_class in ['EXTREME', 'HIGH'] or corr_class == 'HIGH':
            return "DEFENSIVE", max(0.3, 1 - risk_score * 0.15)
        elif vix_class == 'ELEVATED' or corr_class == 'NORMAL_HIGH':
            return "CAUTIOUS", max(0.5, 1 - risk_score * 0.1)
        elif corr_class == 'HIGH':
            return "RISK-OFF", 0.6
        else:
            return "RISK-ON", 1.0

    def compute_position_sizing(self, vol: float, vix: float) -> float:
        """Compute data-driven position sizing factor."""
        vol_class, vol_z = self.classify_volatility(vol)
        vix_class, vix_z = self.classify_vix(vix)

        # Base sizing = 1.0, adjust based on vol and vix z-scores
        base = 1.0

        # Reduce for high volatility
        if vol_z > 0:
            base *= max(0.3, 1 - vol_z * 0.2)

        # Reduce for high VIX
        if vix_z > 0:
            base *= max(0.5, 1 - vix_z * 0.15)

        return round(base, 2)


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
    cursor = conn.execute("SELECT ticker, sector FROM fundamentals_general")
    mapping = {}
    for ticker, sector in cursor:
        if sector:
            mapping[ticker] = sector
    logger.info(f"Loaded sector mapping for {len(mapping)} tickers")
    return mapping


def load_pairwise_correlations(conn: sqlite3.Connection) -> Dict[Tuple[str, str, str], Dict]:
    """Load pairwise correlation and cointegration data."""
    logger.info("Loading pairwise correlations...")
    cursor = conn.execute("""
        SELECT date, ticker1, ticker2, pearson, spearman,
               tail_dep_upper, tail_dep_lower,
               is_cointegrated, spread_zscore, hedge_ratio
        FROM quant_pairwise
    """)
    pairwise = {}
    for row in cursor:
        date = str(row[0])[:10]
        key = (row[1], row[2], date)
        pairwise[key] = {
            'pearson': row[3],
            'spearman': row[4],
            'tail_dep_upper': row[5],
            'tail_dep_lower': row[6],
            'is_cointegrated': row[7],
            'spread_zscore': row[8],
            'hedge_ratio': row[9]
        }
    logger.info(f"Loaded {len(pairwise):,} pairwise records")
    return pairwise


def load_network_data(conn: sqlite3.Connection) -> Dict[str, Dict]:
    """Load network topology data."""
    logger.info("Loading network data...")
    cursor = conn.execute("""
        SELECT date, n_edges, sparsity, hub_nodes
        FROM quant_network
    """)
    networks = {}
    for row in cursor:
        date = str(row[0])[:10]
        networks[date] = {
            'n_edges': row[1],
            'sparsity': row[2],
            'hub_nodes': row[3]
        }
    logger.info(f"Loaded {len(networks):,} network records")
    return networks


# =============================================================================
# Market Outcome Computation
# =============================================================================

def compute_market_outcome(
    prices: Dict[str, Dict],
    ticker: str,
    date: str,
    thresholds: DataDrivenThresholds
) -> Optional[Dict]:
    """Compute market outcome with data-driven direction classification."""
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

    # Data-driven direction classification
    direction = thresholds.classify_return(return_1d)

    # Volatility (next 5 days)
    future_prices = [ticker_prices[dates[i]]['close']
                     for i in range(date_idx, min(date_idx + 6, len(dates)))]
    if len(future_prices) >= 2:
        returns = [(future_prices[i+1] / future_prices[i]) - 1
                   for i in range(len(future_prices) - 1)]
        volatility = sum(r**2 for r in returns) ** 0.5
    else:
        volatility = 0.0

    # Compute z-scores for returns
    ret_thresh = thresholds.thresholds['returns']
    return_zscore = (return_1d - ret_thresh['mean']) / ret_thresh['std'] if ret_thresh['std'] > 0 else 0

    return {
        'return_1d': round(return_1d * 100, 4),
        'return_5d': round(return_5d * 100, 4),
        'return_zscore': round(return_zscore, 2),
        'direction': direction,
        'volatility': round(volatility * 100, 4)
    }


# =============================================================================
# Structured Input Builder - DATA-DRIVEN
# =============================================================================

def build_structured_input(
    ticker: str,
    date: str,
    text: str,
    daily_stats: Dict,
    regimes: Dict,
    sectors: Dict,
    ticker_sectors: Dict,
    pairwise: Dict,
    networks: Dict,
    thresholds: DataDrivenThresholds
) -> str:
    """Build structured input with data-driven classifications."""

    parts = []

    # =========================================================================
    # 1. CONTEXT HEADER
    # =========================================================================
    parts.append(f"<CONTEXT>")
    parts.append(f"TICKER: {ticker}")
    parts.append(f"DATE: {date}")
    parts.append(f"SECTOR: {ticker_sectors.get(ticker, 'Unknown')}")
    parts.append(f"</CONTEXT>")

    # =========================================================================
    # 2. MARKET REGIME (Data-Driven)
    # =========================================================================
    regime_data = regimes.get(date, {})
    if regime_data:
        regime = regime_data.get('regime', 'Normal')
        vix = regime_data.get('vix_level')
        corr = regime_data.get('avg_correlation')
        dispersion = regime_data.get('dispersion')
        hmm = regime_data.get('hmm_state', 0)

        # Data-driven classifications
        vix_class, vix_z = thresholds.classify_vix(vix)
        corr_class, corr_z = thresholds.classify_correlation(corr)
        mode, risk_budget = thresholds.get_regime_mode(vix, corr)

        parts.append(f"\n<REGIME>")
        parts.append(f"MARKET_STATE: {regime}")
        parts.append(f"MODE: {mode}")
        parts.append(f"RISK_BUDGET: {risk_budget:.0%}")

        if vix is not None:
            parts.append(f"VIX: {vix:.1f} [{vix_class}] (z={vix_z:+.2f})")
        if corr is not None:
            parts.append(f"CORRELATION: {corr:.3f} [{corr_class}] (z={corr_z:+.2f})")
        if dispersion is not None:
            parts.append(f"DISPERSION: {dispersion:.3f}")

        parts.append(f"HMM_STATE: {hmm} ({'Stress' if hmm == 1 else 'Normal'})")
        parts.append(f"</REGIME>")

    # =========================================================================
    # 3. TICKER STATISTICS (Data-Driven)
    # =========================================================================
    stats = daily_stats.get((ticker, date), {})
    if stats:
        momentum = stats.get('momentum_20d')
        vol = stats.get('std_20d')
        skew = stats.get('skew_20d')
        kurt = stats.get('kurt_20d')
        var = stats.get('var_5pct')

        # Data-driven classifications
        mom_class, mom_z = thresholds.classify_momentum(momentum)
        vol_class, vol_z = thresholds.classify_volatility(vol)
        skew_class, skew_z = thresholds.classify_skewness(skew)
        kurt_class, kurt_z = thresholds.classify_kurtosis(kurt)

        parts.append(f"\n<TICKER_STATS>")

        if momentum is not None:
            parts.append(f"MOMENTUM_20D: {momentum*100:+.2f}% [{mom_class}] (z={mom_z:+.2f})")

        if vol is not None:
            parts.append(f"VOLATILITY_20D: {vol*100:.2f}% [{vol_class}] (z={vol_z:+.2f})")

        if skew is not None:
            parts.append(f"SKEWNESS: {skew:+.2f} [{skew_class}] (z={skew_z:+.2f})")
            if skew_class in ['STRONG_LEFT_TAIL', 'LEFT_TAIL']:
                parts.append(f"  [!] Crash probability elevated vs historical norm")

        if kurt is not None:
            parts.append(f"KURTOSIS: {kurt:.2f} [{kurt_class}] (z={kurt_z:+.2f})")
            if kurt_class in ['FAT_TAILS', 'EXTREME_TAILS']:
                parts.append(f"  [!] Extreme moves more likely than normal")

        if var is not None:
            parts.append(f"VAR_5PCT: {var*100:.2f}%")

        parts.append(f"</TICKER_STATS>")

    # =========================================================================
    # 4. CORRELATION NETWORK
    # =========================================================================
    correlations = []
    for (t1, t2, d), data in pairwise.items():
        if d == date and (t1 == ticker or t2 == ticker):
            other = t2 if t1 == ticker else t1
            correlations.append({
                'ticker': other,
                'pearson': data.get('pearson', 0),
                'spearman': data.get('spearman', 0),
                'tail_up': data.get('tail_dep_upper', 0),
                'tail_dn': data.get('tail_dep_lower', 0),
                'coint': data.get('is_cointegrated', False),
                'zscore': data.get('spread_zscore', 0),
                'hedge': data.get('hedge_ratio', 1)
            })

    correlations.sort(key=lambda x: abs(x.get('pearson', 0) or 0), reverse=True)

    if correlations:
        parts.append(f"\n<CORRELATIONS>")
        parts.append(f"TOP CORRELATED ASSETS:")
        for c in correlations[:5]:
            pearson = c.get('pearson', 0) or 0
            parts.append(f"  {ticker}-{c['ticker']}: r={pearson:+.3f}")
        parts.append(f"</CORRELATIONS>")

    # =========================================================================
    # 5. TAIL DEPENDENCIES (Data-Driven)
    # =========================================================================
    tail_risks = []
    for c in correlations:
        tail_dn = c.get('tail_dn') or 0
        tail_up = c.get('tail_up') or 0
        dn_class, _ = thresholds.classify_tail_dep(tail_dn, 'lower')
        up_class, _ = thresholds.classify_tail_dep(tail_up, 'upper')

        if dn_class in ['MODERATE', 'HIGH'] or up_class in ['MODERATE', 'HIGH']:
            tail_risks.append({
                'ticker': c['ticker'],
                'tail_dn': tail_dn,
                'tail_up': tail_up,
                'dn_class': dn_class,
                'up_class': up_class
            })

    if tail_risks:
        parts.append(f"\n<TAIL_DEPENDENCIES>")
        parts.append(f"[!] CRASH/RALLY CORRELATIONS DETECTED:")
        for tr in tail_risks[:5]:
            parts.append(f"  {ticker}-{tr['ticker']}: crash_corr={tr['tail_dn']:.3f} [{tr['dn_class']}], rally_corr={tr['tail_up']:.3f} [{tr['up_class']}]")
            if tr['dn_class'] == 'HIGH':
                parts.append(f"    [!] HIGH CRASH CORRELATION vs historical distribution")
        parts.append(f"</TAIL_DEPENDENCIES>")

    # =========================================================================
    # 6. COINTEGRATION OPPORTUNITIES (Data-Driven)
    # =========================================================================
    coint_pairs = [c for c in correlations if c.get('coint')]
    coint_signals = []
    for c in coint_pairs:
        zscore_val = c.get('zscore', 0) or 0
        signal_class, signal_strength = thresholds.classify_zscore(zscore_val)
        if signal_class in ['MODERATE', 'STRONG', 'EXTREME']:
            coint_signals.append({
                **c,
                'signal_class': signal_class,
                'signal_strength': signal_strength
            })

    if coint_signals:
        parts.append(f"\n<COINTEGRATION>")
        parts.append(f"[PAIRS TRADING OPPORTUNITIES]:")
        for c in coint_signals[:3]:
            zscore = c.get('zscore', 0) or 0
            hedge = c.get('hedge', 1) or 1

            if zscore < 0:
                signal = "LONG SPREAD"
                action = f"BUY {ticker}, SELL {hedge:.2f}x {c['ticker']}"
            else:
                signal = "SHORT SPREAD"
                action = f"SELL {ticker}, BUY {hedge:.2f}x {c['ticker']}"

            parts.append(f"  {ticker}/{c['ticker']}: z={zscore:+.2f} [{c['signal_class']}]")
            parts.append(f"    Signal: {signal}")
            parts.append(f"    Action: {action}")
            parts.append(f"    Strength: {c['signal_strength']:.2f} std from mean")
        parts.append(f"</COINTEGRATION>")

    # =========================================================================
    # 7. SECTOR CONTEXT
    # =========================================================================
    sector = ticker_sectors.get(ticker)
    if sector and (sector, date) in sectors:
        sec = sectors[(sector, date)]
        sec_ret_1d = sec.get('sector_return_1d') or 0
        sec_ret_20d = sec.get('sector_return_20d') or 0
        sec_vol = sec.get('sector_volatility') or 0
        sec_beta = sec.get('sector_beta') or 1
        sec_corr = sec.get('sector_internal_corr') or 0

        parts.append(f"\n<SECTOR_CONTEXT>")
        parts.append(f"SECTOR: {sector}")
        parts.append(f"RETURN_1D: {sec_ret_1d*100:+.2f}%")
        parts.append(f"RETURN_20D: {sec_ret_20d*100:+.2f}%")
        parts.append(f"VOLATILITY: {sec_vol*100:.2f}%")
        parts.append(f"BETA: {sec_beta:.2f}")
        parts.append(f"INTERNAL_CORR: {sec_corr:.3f}")
        parts.append(f"</SECTOR_CONTEXT>")

    # =========================================================================
    # 8. NEWS/EVENT
    # =========================================================================
    parts.append(f"\n<NEWS>")
    parts.append(text[:4000])
    parts.append(f"</NEWS>")

    return "\n".join(parts)


# =============================================================================
# Semantic Analysis Output Builder - DATA-DRIVEN
# =============================================================================

def build_semantic_output(
    ticker: str,
    date: str,
    outcome: Dict,
    daily_stats: Dict,
    regimes: Dict,
    sectors: Dict,
    ticker_sectors: Dict,
    pairwise: Dict,
    thresholds: DataDrivenThresholds
) -> str:
    """Build semantic analysis output with data-driven classifications."""

    ret_1d = outcome['return_1d']
    ret_5d = outcome['return_5d']
    ret_zscore = outcome.get('return_zscore', 0)
    direction = outcome['direction']
    volatility = outcome['volatility']

    # Get context data
    regime_data = regimes.get(date, {})
    stats = daily_stats.get((ticker, date), {})
    sector = ticker_sectors.get(ticker, 'Unknown')

    vix = regime_data.get('vix_level')
    corr = regime_data.get('avg_correlation')
    momentum = stats.get('momentum_20d')
    vol = stats.get('std_20d')
    skew = stats.get('skew_20d')
    kurt = stats.get('kurt_20d')

    # Data-driven classifications
    vix_class, vix_z = thresholds.classify_vix(vix) if vix else ("UNKNOWN", 0)
    corr_class, corr_z = thresholds.classify_correlation(corr) if corr else ("UNKNOWN", 0)
    mom_class, mom_z = thresholds.classify_momentum(momentum) if momentum else ("NEUTRAL", 0)
    vol_class, vol_z = thresholds.classify_volatility(vol) if vol else ("UNKNOWN", 0)
    skew_class, skew_z = thresholds.classify_skewness(skew) if skew else ("NORMAL", 0)
    kurt_class, kurt_z = thresholds.classify_kurtosis(kurt) if kurt else ("NORMAL", 0)
    mode, risk_budget = thresholds.get_regime_mode(vix, corr) if vix and corr else ("RISK-ON", 1.0)

    # Position sizing
    sizing_factor = thresholds.compute_position_sizing(vol, vix) if vol and vix else 1.0

    # Find cointegrated pairs with signals
    coint_opps = []
    tail_risks = []
    for (t1, t2, d), data in pairwise.items():
        if d == date and (t1 == ticker or t2 == ticker):
            other = t2 if t1 == ticker else t1
            zscore_val = data.get('spread_zscore', 0) or 0
            signal_class, _ = thresholds.classify_zscore(zscore_val)

            if data.get('is_cointegrated') and signal_class in ['MODERATE', 'STRONG', 'EXTREME']:
                coint_opps.append({
                    'ticker': other,
                    'zscore': zscore_val,
                    'hedge': data.get('hedge_ratio', 1),
                    'signal_class': signal_class
                })

            tail_dn = data.get('tail_dep_lower', 0) or 0
            dn_class, _ = thresholds.classify_tail_dep(tail_dn, 'lower')
            if dn_class in ['MODERATE', 'HIGH']:
                tail_risks.append({'ticker': other, 'tail': tail_dn, 'class': dn_class})

    # Build output
    output = []

    # =========================================================================
    # COMPREHENSION
    # =========================================================================
    output.append("<COMPREHENSION>")
    output.append(f"Market in {mode} mode (risk_budget={risk_budget:.0%}).")
    if vix:
        output.append(f"VIX at {vix:.1f} is {vix_class} (z={vix_z:+.2f} vs historical).")
    if corr:
        output.append(f"Market correlation at {corr:.2f} is {corr_class} (z={corr_z:+.2f}).")
    output.append("</COMPREHENSION>")

    # =========================================================================
    # QUANTITATIVE READING
    # =========================================================================
    output.append("\n<QUANTITATIVE_READING>")
    if momentum is not None:
        output.append(f"MOMENTUM: {momentum*100:+.1f}% [{mom_class}] (z={mom_z:+.2f})")
        if mom_class in ['STRONG_UP', 'UP']:
            output.append(f"  -> Trend following signal (above p{90 if mom_class == 'UP' else 95} of distribution)")
        elif mom_class in ['STRONG_DOWN', 'DOWN']:
            output.append(f"  -> Bearish trend (below p{10 if mom_class == 'DOWN' else 5} of distribution)")

    if vol is not None:
        output.append(f"VOLATILITY: {vol*100:.1f}% [{vol_class}] (z={vol_z:+.2f})")
        if vol_class in ['HIGH', 'EXTREME']:
            reduction = int((1 - sizing_factor) * 100)
            output.append(f"  -> Reduce position sizing by {reduction}% (data-driven)")

    if skew is not None:
        output.append(f"SKEWNESS: {skew:+.2f} [{skew_class}] (z={skew_z:+.2f})")
        if skew_class in ['STRONG_LEFT_TAIL', 'LEFT_TAIL']:
            output.append(f"  -> Left tail risk elevated vs historical norm")

    if kurt is not None:
        output.append(f"KURTOSIS: {kurt:.1f} [{kurt_class}] (z={kurt_z:+.2f})")
        if kurt_class in ['FAT_TAILS', 'EXTREME_TAILS']:
            output.append(f"  -> VaR understates true risk (fatter tails than normal)")
    output.append("</QUANTITATIVE_READING>")

    # =========================================================================
    # CORRELATION INSIGHT
    # =========================================================================
    output.append("\n<CORRELATION_INSIGHT>")
    if tail_risks:
        output.append("TAIL RISK CLUSTERS DETECTED:")
        for tr in tail_risks[:3]:
            output.append(f"  {ticker} crashes with {tr['ticker']} (lambda={tr['tail']:.2f}, {tr['class']})")
        output.append("  -> Diversification effectiveness reduced")
    else:
        output.append("No significant tail dependencies detected")
        output.append("Standard diversification should be effective")
    output.append("</CORRELATION_INSIGHT>")

    # =========================================================================
    # SYNTHESIS
    # =========================================================================
    output.append("\n<SYNTHESIS>")

    # Confidence based on return z-score
    if abs(ret_zscore) > 2:
        confidence = "HIGH"
    elif abs(ret_zscore) > 1:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    if direction == 'up':
        if mom_class in ['UP', 'STRONG_UP', 'WEAK_UP']:
            output.append(f"BULLISH: Momentum + catalyst alignment")
        else:
            output.append(f"BULLISH REVERSAL: Event-driven upside against weak momentum")
    elif direction == 'down':
        if mom_class in ['DOWN', 'STRONG_DOWN', 'WEAK_DOWN']:
            output.append(f"BEARISH: Momentum + catalyst alignment")
        else:
            output.append(f"BEARISH REVERSAL: Event-driven downside against positive momentum")
    else:
        output.append(f"NEUTRAL: No strong directional signal")

    output.append(f"EXPECTED_MOVE: {ret_1d:+.2f}% (1d), {ret_5d:+.2f}% (5d)")
    output.append(f"RETURN_ZSCORE: {ret_zscore:+.2f}")
    output.append(f"CONFIDENCE: {confidence}")
    output.append("</SYNTHESIS>")

    # =========================================================================
    # OPPORTUNITIES
    # =========================================================================
    output.append("\n<OPPORTUNITIES>")

    # Momentum opportunity
    if mom_class in ['STRONG_UP', 'UP']:
        output.append(f"[MOMENTUM] LONG {ticker}")
        output.append(f"  Trend: {mom_class} (z={mom_z:+.2f})")
    elif mom_class in ['STRONG_DOWN', 'DOWN']:
        output.append(f"[MOMENTUM] SHORT {ticker}")
        output.append(f"  Trend: {mom_class} (z={mom_z:+.2f})")

    # Mean reversion (based on return z-score)
    if ret_zscore < -2:
        output.append(f"[MEAN_REVERSION] {ticker} oversold (z={ret_zscore:.2f})")
        output.append(f"  Drop: {ret_1d:.1f}% (>2 std below mean)")

    # Pairs trading
    for opp in coint_opps[:2]:
        zscore = opp['zscore'] or 0
        hedge = opp['hedge'] or 1
        if zscore < 0:
            output.append(f"[PAIRS] LONG SPREAD {ticker}/{opp['ticker']}")
        else:
            output.append(f"[PAIRS] SHORT SPREAD {ticker}/{opp['ticker']}")
        output.append(f"  Z-score: {zscore:+.2f} [{opp['signal_class']}]")
        output.append(f"  Hedge: {hedge:.2f}")

    if not coint_opps and mom_class == 'NEUTRAL' and abs(ret_zscore) < 1:
        output.append("No strong opportunities detected")
        output.append("Wait for clearer signals")

    output.append("</OPPORTUNITIES>")

    # =========================================================================
    # RISKS
    # =========================================================================
    output.append("\n<RISKS>")

    # Compute overall risk level from z-scores
    risk_factors = []
    if vix_z > 1:
        risk_factors.append(f"VIX elevated (z={vix_z:+.2f})")
    if vol_z > 1:
        risk_factors.append(f"High volatility (z={vol_z:+.2f})")
    if skew_z < -1:
        risk_factors.append(f"Left tail risk (z={skew_z:+.2f})")
    if kurt_z > 1:
        risk_factors.append(f"Fat tails (z={kurt_z:+.2f})")
    if tail_risks:
        risk_factors.append(f"{len(tail_risks)} crash-correlated assets")

    if len(risk_factors) >= 3:
        risk_level = "HIGH"
    elif len(risk_factors) >= 1:
        risk_level = "ELEVATED"
    else:
        risk_level = "NORMAL"

    output.append(f"OVERALL_RISK: {risk_level}")
    for rf in risk_factors:
        output.append(f"  [!] {rf}")

    if risk_level == "NORMAL":
        output.append("  Standard risk management sufficient")

    output.append("</RISKS>")

    # =========================================================================
    # PORTFOLIO ACTION
    # =========================================================================
    output.append("\n<ACTION>")

    # Data-driven position sizing
    base_size = 0.02  # 2% base
    adjusted_size = base_size * sizing_factor * risk_budget

    if direction == 'up' and abs(ret_zscore) > 1:
        output.append(f"LONG {ticker}")
        output.append(f"SIZING: {adjusted_size:.1%} of portfolio (base={base_size:.0%} x sizing={sizing_factor:.2f} x risk_budget={risk_budget:.2f})")
        if vol:
            stop_distance = vol * 100 * 2  # 2x daily vol
            output.append(f"STOP: {stop_distance:.1f}% below entry (2x daily vol)")
    elif direction == 'down' and abs(ret_zscore) > 1:
        output.append(f"SHORT/UNDERWEIGHT {ticker}")
        output.append(f"SIZING: {adjusted_size:.1%} reduction")
    else:
        output.append(f"HOLD / NEUTRAL on {ticker}")
        output.append(f"No strong directional signal (|z|={abs(ret_zscore):.2f} < 1)")

    if risk_budget < 1:
        output.append(f"\n[RISK_BUDGET] All sizes reduced to {risk_budget:.0%} of normal (regime-driven)")

    output.append("</ACTION>")

    # =========================================================================
    # SAC VECTOR
    # =========================================================================
    output.append("\n<SAC_VECTOR>")
    output.append(f"regime_mode: {mode}")
    output.append(f"risk_budget: {risk_budget:.2f}")
    output.append(f"vix: {vix if vix else 0}")
    output.append(f"vix_zscore: {vix_z:.2f}")
    output.append(f"correlation: {corr if corr else 0}")
    output.append(f"correlation_zscore: {corr_z:.2f}")
    output.append(f"momentum: {momentum if momentum else 0}")
    output.append(f"momentum_zscore: {mom_z:.2f}")
    output.append(f"volatility: {vol if vol else 0}")
    output.append(f"volatility_zscore: {vol_z:.2f}")
    output.append(f"skewness_zscore: {skew_z:.2f}")
    output.append(f"kurtosis_zscore: {kurt_z:.2f}")
    output.append(f"direction_signal: {1 if direction == 'up' else -1 if direction == 'down' else 0}")
    output.append(f"return_zscore: {ret_zscore:.2f}")
    output.append(f"expected_return_1d: {ret_1d:.4f}")
    output.append(f"expected_return_5d: {ret_5d:.4f}")
    output.append(f"confidence: {confidence.lower()}")
    output.append(f"sizing_factor: {sizing_factor:.2f}")
    output.append(f"pairs_opportunities: {len(coint_opps)}")
    output.append(f"tail_risk_count: {len(tail_risks)}")
    output.append(f"risk_level: {risk_level.lower()}")
    output.append("</SAC_VECTOR>")

    return "\n".join(output)


# =============================================================================
# Training Sample Creation
# =============================================================================

def create_training_sample(
    source_type: str,
    ticker: str,
    date: str,
    text: str,
    outcome: Dict,
    structured_input: str,
    semantic_output: str
) -> Dict:
    """Create final training sample."""

    return {
        'source_type': source_type,
        'ticker': ticker,
        'date': date,
        'input': structured_input,
        'output': semantic_output,
        'ground_truth': outcome,
        'metadata': {
            'original_text_length': len(text),
            'input_length': len(structured_input),
            'output_length': len(semantic_output)
        }
    }


# =============================================================================
# Processing Functions
# =============================================================================

def process_news(
    conn: sqlite3.Connection,
    prices: Dict,
    daily_stats: Dict,
    regimes: Dict,
    sectors: Dict,
    ticker_sectors: Dict,
    pairwise: Dict,
    networks: Dict,
    thresholds: DataDrivenThresholds,
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
        outcome = compute_market_outcome(prices, ticker, date, thresholds)
        if outcome is None:
            skipped += 1
            continue

        text = title
        if content:
            text += f". {content[:3000]}"

        # Build structured input
        structured_input = build_structured_input(
            ticker, date, text,
            daily_stats, regimes, sectors, ticker_sectors, pairwise, networks,
            thresholds
        )

        # Build semantic output
        semantic_output = build_semantic_output(
            ticker, date, outcome,
            daily_stats, regimes, sectors, ticker_sectors, pairwise,
            thresholds
        )

        sample = create_training_sample(
            source_type='news',
            ticker=ticker,
            date=date,
            text=text,
            outcome=outcome,
            structured_input=structured_input,
            semantic_output=semantic_output
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
    pairwise: Dict,
    networks: Dict,
    thresholds: DataDrivenThresholds,
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

        date = None
        if filing_date:
            date = str(filing_date)[:10]
        else:
            match = re.search(r'FILED AS OF DATE:\s*(\d{8})', content)
            if match:
                d = match.group(1)
                date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"

        if not date:
            skipped += 1
            continue

        outcome = compute_market_outcome(prices, ticker, date, thresholds)
        if outcome is None:
            skipped += 1
            continue

        # Extract key sections
        text_parts = []
        content_lower = content.lower()

        if 'item 1.' in content_lower:
            start = content_lower.find('item 1.')
            end = content_lower.find('item 2.', start) if start > 0 else -1
            if start > 0 and end > start:
                text_parts.append(content[start:min(end, start+5000)])

        if 'risk factors' in content_lower:
            start = content_lower.find('risk factors')
            text_parts.append(content[start:start+3000])

        if not text_parts:
            text_parts.append(content[:6000])

        text = f"SEC 10-K Filing for {ticker}. " + " ".join(text_parts)

        structured_input = build_structured_input(
            ticker, date, text,
            daily_stats, regimes, sectors, ticker_sectors, pairwise, networks,
            thresholds
        )

        semantic_output = build_semantic_output(
            ticker, date, outcome,
            daily_stats, regimes, sectors, ticker_sectors, pairwise,
            thresholds
        )

        sample = create_training_sample(
            source_type='sec_10k',
            ticker=ticker,
            date=date,
            text=text,
            outcome=outcome,
            structured_input=structured_input,
            semantic_output=semantic_output
        )
        samples.append(sample)

    logger.info(f"Processed {len(samples):,} SEC samples (skipped {skipped:,})")
    return samples


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare DATA-DRIVEN training data for Promethee")
    parser.add_argument("--db_path", type=str, required=True, help="Path to SQLite database")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--max_news", type=int, default=250000, help="Max news samples")
    parser.add_argument("--max_sec", type=int, default=50000, help="Max SEC samples")
    parser.add_argument("--start_date", type=str, default="2010-01-01", help="Start date")
    parser.add_argument("--end_date", type=str, default="2025-12-31", help="End date")
    parser.add_argument("--news_sample_rate", type=float, default=0.2, help="News sampling rate")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output")
    parser.add_argument("--save_thresholds", type=str, default=None, help="Save computed thresholds to JSON")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PROMETHEE - DATA-DRIVEN Training Data Preparation")
    logger.info("=" * 70)
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info("")

    # Connect
    conn = sqlite3.connect(args.db_path)

    # Compute data-driven thresholds FIRST
    thresholds = DataDrivenThresholds()
    thresholds.compute_from_data(conn)

    # Optionally save thresholds
    if args.save_thresholds:
        with open(args.save_thresholds, 'w') as f:
            json.dump(thresholds.thresholds, f, indent=2)
        logger.info(f"Thresholds saved to: {args.save_thresholds}")

    logger.info("")

    # Load all reference data
    prices = load_prices(conn)
    daily_stats = load_quant_daily_stats(conn)
    regimes = load_quant_regime(conn)
    sectors = load_quant_sector(conn)
    ticker_sectors = load_ticker_sectors(conn)
    pairwise = load_pairwise_correlations(conn)
    networks = load_network_data(conn)

    logger.info("")

    # Process news
    news_samples = process_news(
        conn, prices, daily_stats, regimes, sectors, ticker_sectors,
        pairwise, networks, thresholds,
        args.start_date, args.end_date, args.max_news, args.news_sample_rate
    )

    # Process SEC
    sec_samples = process_sec_filings(
        conn, prices, daily_stats, regimes, sectors, ticker_sectors,
        pairwise, networks, thresholds, args.max_sec
    )

    conn.close()

    # Combine
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

    # Direction distribution
    directions = defaultdict(int)
    for s in all_samples:
        directions[s['ground_truth']['direction']] += 1

    logger.info("")
    logger.info("Direction distribution:")
    for d, count in sorted(directions.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_samples)
        logger.info(f"  {d}: {count:,} ({pct:.1f}%)")

    logger.info("")
    logger.info(f"Output saved to: {output_path}")

    # Show sample
    if all_samples:
        logger.info("")
        logger.info("=" * 70)
        logger.info("SAMPLE OUTPUT")
        logger.info("=" * 70)
        sample = all_samples[0]
        logger.info(f"Source: {sample['source_type']}")
        logger.info(f"Ticker: {sample['ticker']}, Date: {sample['date']}")
        logger.info("")
        logger.info("--- INPUT (first 600 chars) ---")
        logger.info(sample['input'][:600])
        logger.info("")
        logger.info("--- OUTPUT (first 600 chars) ---")
        logger.info(sample['output'][:600])


if __name__ == "__main__":
    main()
