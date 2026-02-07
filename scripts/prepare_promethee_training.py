"""
Prepare SOTA Training Data for Promethee

COMPREHENSIVE multi-modal training data with:
1. Structured context input (regime, correlations, tail deps, cointegration)
2. Semantic analysis output (opportunities, risks, synthesis)
3. SAC vector output for reinforcement learning agent

This teaches Promethee to:
- Understand market structure and dependencies
- Detect trading opportunities (pairs, momentum, mean reversion)
- Assess tail risks and concentration
- Output embeddings for SAC agent

Usage:
    python scripts/prepare_promethee_training.py \
        --db_path eodhd_sp500.db \
        --output_path data/promethee_semantic_training.jsonl \
        --max_samples 300000
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import random
import re

from tqdm import tqdm
from loguru import logger


# =============================================================================
# Data Loaders - Extended for Semantic Analysis
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


def load_pairwise_correlations(conn: sqlite3.Connection) -> Dict[Tuple[str, str], Dict]:
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

def compute_market_outcome(prices: Dict[str, Dict], ticker: str, date: str) -> Optional[Dict]:
    """Compute market outcome for a given (ticker, date)."""
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
# Structured Input Builder - SEMANTIC FORMAT
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
    networks: Dict
) -> str:
    """Build structured semantic input for Promethee."""

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
    # 2. MARKET REGIME
    # =========================================================================
    regime_data = regimes.get(date, {})
    if regime_data:
        regime = regime_data.get('regime', 'Normal')
        vix = regime_data.get('vix_level', 18)
        corr = regime_data.get('avg_correlation', 0.3)
        dispersion = regime_data.get('dispersion', 0)
        hmm = regime_data.get('hmm_state', 0)

        # Determine mode
        if vix and vix > 25:
            mode = "DEFENSIVE"
        elif vix and vix > 20:
            mode = "CAUTIOUS"
        elif corr and corr > 0.6:
            mode = "RISK-OFF"
        else:
            mode = "RISK-ON"

        parts.append(f"\n<REGIME>")
        parts.append(f"MARKET_STATE: {regime}")
        parts.append(f"MODE: {mode}")
        parts.append(f"VIX: {vix:.1f}" if vix else "VIX: N/A")
        parts.append(f"CORRELATION: {corr:.3f}" if corr else "CORRELATION: N/A")
        parts.append(f"DISPERSION: {dispersion:.3f}" if dispersion else "DISPERSION: N/A")
        parts.append(f"HMM_STATE: {hmm} ({'Stress' if hmm == 1 else 'Normal'})")
        parts.append(f"</REGIME>")

    # =========================================================================
    # 3. TICKER STATISTICS
    # =========================================================================
    stats = daily_stats.get((ticker, date), {})
    if stats:
        momentum = stats.get('momentum_20d', 0)
        vol = stats.get('std_20d', 0)
        skew = stats.get('skew_20d', 0)
        kurt = stats.get('kurt_20d', 0)
        var = stats.get('var_5pct', 0)

        parts.append(f"\n<TICKER_STATS>")
        parts.append(f"MOMENTUM_20D: {momentum*100:+.2f}%" if momentum else "MOMENTUM_20D: N/A")

        # Momentum flags
        if momentum:
            if abs(momentum) > 0.15:
                parts.append(f"  [!] EXTREME MOMENTUM (>15%)")
            elif abs(momentum) > 0.08:
                parts.append(f"  [!] STRONG {'UP' if momentum > 0 else 'DOWN'}TREND")

        parts.append(f"VOLATILITY_20D: {vol*100:.2f}%" if vol else "VOLATILITY_20D: N/A")
        if vol and vol > 0.04:
            parts.append(f"  [!] HIGH VOLATILITY")

        parts.append(f"SKEWNESS: {skew:+.2f}" if skew else "SKEWNESS: N/A")
        if skew and skew < -1:
            parts.append(f"  [!] LEFT TAIL RISK - Crash probability elevated")
        elif skew and skew > 1:
            parts.append(f"  [+] RIGHT SKEW - Upside potential")

        parts.append(f"KURTOSIS: {kurt:.2f}" if kurt else "KURTOSIS: N/A")
        if kurt and kurt > 5:
            parts.append(f"  [!] FAT TAILS - Extreme moves more likely than normal")

        parts.append(f"VAR_5PCT: {var*100:.2f}%" if var else "VAR_5PCT: N/A")
        parts.append(f"</TICKER_STATS>")

    # =========================================================================
    # 4. CORRELATION NETWORK
    # =========================================================================
    # Find correlations for this ticker
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

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x.get('pearson', 0) or 0), reverse=True)

    if correlations:
        parts.append(f"\n<CORRELATIONS>")
        parts.append(f"TOP CORRELATED ASSETS:")
        for c in correlations[:5]:
            pearson = c.get('pearson', 0) or 0
            parts.append(f"  {ticker}-{c['ticker']}: r={pearson:+.3f}")
        parts.append(f"</CORRELATIONS>")

    # =========================================================================
    # 5. TAIL DEPENDENCIES
    # =========================================================================
    tail_risks = [c for c in correlations if (c.get('tail_dn') or 0) > 0.5 or (c.get('tail_up') or 0) > 0.5]

    if tail_risks:
        parts.append(f"\n<TAIL_DEPENDENCIES>")
        parts.append(f"[!] CRASH/RALLY CORRELATIONS DETECTED:")
        for c in tail_risks[:5]:
            tail_dn = c.get('tail_dn', 0) or 0
            tail_up = c.get('tail_up', 0) or 0
            parts.append(f"  {ticker}-{c['ticker']}: crash_corr={tail_dn:.3f}, rally_corr={tail_up:.3f}")
            if tail_dn > 0.7:
                parts.append(f"    [!] HIGH CRASH CORRELATION - Diversification ineffective!")
        parts.append(f"</TAIL_DEPENDENCIES>")

    # =========================================================================
    # 6. COINTEGRATION OPPORTUNITIES
    # =========================================================================
    coint_pairs = [c for c in correlations if c.get('coint')]
    coint_signals = [c for c in coint_pairs if abs(c.get('zscore', 0) or 0) > 1.5]

    if coint_signals:
        parts.append(f"\n<COINTEGRATION>")
        parts.append(f"[PAIRS TRADING OPPORTUNITIES]:")
        for c in coint_signals[:3]:
            zscore = c.get('zscore', 0) or 0
            hedge = c.get('hedge', 1) or 1
            if zscore < -2:
                signal = "LONG SPREAD"
                action = f"BUY {ticker}, SELL {hedge:.2f}x {c['ticker']}"
            elif zscore > 2:
                signal = "SHORT SPREAD"
                action = f"SELL {ticker}, BUY {hedge:.2f}x {c['ticker']}"
            else:
                signal = "MODERATE"
                action = f"Watch for z>2 entry"

            parts.append(f"  {ticker}/{c['ticker']}: z={zscore:+.2f}")
            parts.append(f"    Signal: {signal}")
            parts.append(f"    Action: {action}")
            parts.append(f"    Expected reversion: {abs(zscore) * 0.5:.1f}%")
        parts.append(f"</COINTEGRATION>")

    # =========================================================================
    # 7. SECTOR CONTEXT
    # =========================================================================
    sector = ticker_sectors.get(ticker)
    if sector and (sector, date) in sectors:
        sec = sectors[(sector, date)]
        sec_ret_1d = sec.get('sector_return_1d', 0) or 0
        sec_ret_20d = sec.get('sector_return_20d', 0) or 0
        sec_vol = sec.get('sector_volatility', 0) or 0
        sec_beta = sec.get('sector_beta', 1) or 1
        sec_corr = sec.get('sector_internal_corr', 0) or 0

        parts.append(f"\n<SECTOR_CONTEXT>")
        parts.append(f"SECTOR: {sector}")
        parts.append(f"RETURN_1D: {sec_ret_1d*100:+.2f}%")
        parts.append(f"RETURN_20D: {sec_ret_20d*100:+.2f}%")
        parts.append(f"VOLATILITY: {sec_vol*100:.2f}%")
        parts.append(f"BETA: {sec_beta:.2f}")
        parts.append(f"INTERNAL_CORR: {sec_corr:.3f}")

        if sec_corr > 0.7:
            parts.append(f"  [!] HIGH INTERNAL CORRELATION - Sector moves together")
        if sec_beta > 1.3:
            parts.append(f"  [!] HIGH BETA SECTOR - Amplifies market moves")
        if sec_ret_20d > 0.05:
            parts.append(f"  [+] STRONG SECTOR - Overweight bias")
        elif sec_ret_20d < -0.05:
            parts.append(f"  [-] WEAK SECTOR - Underweight bias")
        parts.append(f"</SECTOR_CONTEXT>")

    # =========================================================================
    # 8. NETWORK TOPOLOGY
    # =========================================================================
    network = networks.get(date, {})
    if network:
        n_edges = network.get('n_edges', 0)
        sparsity = network.get('sparsity', 0)
        hubs = network.get('hub_nodes', '')

        parts.append(f"\n<NETWORK>")
        parts.append(f"EDGES: {n_edges}")
        parts.append(f"SPARSITY: {sparsity:.3f}")
        if hubs and ticker in str(hubs):
            parts.append(f"[!] {ticker} is a HUB NODE - Central to market structure")
        parts.append(f"</NETWORK>")

    # =========================================================================
    # 9. NEWS/EVENT
    # =========================================================================
    parts.append(f"\n<NEWS>")
    parts.append(text[:4000])
    parts.append(f"</NEWS>")

    return "\n".join(parts)


# =============================================================================
# Semantic Analysis Output Builder
# =============================================================================

def build_semantic_output(
    ticker: str,
    date: str,
    outcome: Dict,
    daily_stats: Dict,
    regimes: Dict,
    sectors: Dict,
    ticker_sectors: Dict,
    pairwise: Dict
) -> str:
    """Build semantic analysis output that Promethee should produce."""

    ret_1d = outcome['return_1d']
    ret_5d = outcome['return_5d']
    direction = outcome['direction']
    volatility = outcome['volatility']

    # Get context data
    regime_data = regimes.get(date, {})
    stats = daily_stats.get((ticker, date), {})
    sector = ticker_sectors.get(ticker, 'Unknown')

    vix = regime_data.get('vix_level', 18) or 18
    corr = regime_data.get('avg_correlation', 0.3) or 0.3
    momentum = stats.get('momentum_20d', 0) or 0
    vol = stats.get('std_20d', 0.02) or 0.02
    skew = stats.get('skew_20d', 0) or 0
    kurt = stats.get('kurt_20d', 3) or 3

    # Find cointegrated pairs with signals
    coint_opps = []
    tail_risks = []
    for (t1, t2, d), data in pairwise.items():
        if d == date and (t1 == ticker or t2 == ticker):
            other = t2 if t1 == ticker else t1
            if data.get('is_cointegrated') and abs(data.get('spread_zscore', 0) or 0) > 1.5:
                coint_opps.append({
                    'ticker': other,
                    'zscore': data.get('spread_zscore', 0),
                    'hedge': data.get('hedge_ratio', 1)
                })
            if (data.get('tail_dep_lower', 0) or 0) > 0.5:
                tail_risks.append({'ticker': other, 'tail': data.get('tail_dep_lower', 0)})

    # Determine market mode
    if vix > 25:
        mode = "DEFENSIVE"
        risk_budget = 0.5
    elif vix > 20:
        mode = "CAUTIOUS"
        risk_budget = 0.75
    elif corr > 0.6:
        mode = "RISK-OFF"
        risk_budget = 0.6
    else:
        mode = "RISK-ON"
        risk_budget = 1.0

    # Build output
    output = []

    # =========================================================================
    # COMPREHENSION
    # =========================================================================
    output.append("<COMPREHENSION>")
    output.append(f"I analyzed {ticker} in a {mode} market environment.")
    output.append(f"VIX at {vix:.1f} indicates {'elevated fear' if vix > 22 else 'normal risk appetite'}.")
    output.append(f"Market correlation at {corr:.2f} suggests {'macro-driven moves' if corr > 0.5 else 'stock-picking opportunity'}.")
    output.append("</COMPREHENSION>")

    # =========================================================================
    # QUANTITATIVE READING
    # =========================================================================
    output.append("\n<QUANTITATIVE_READING>")
    output.append(f"MOMENTUM: {momentum*100:+.1f}% over 20 days")
    if abs(momentum) > 0.10:
        output.append(f"  -> STRONG {'UP' if momentum > 0 else 'DOWN'}TREND - Trend following signal")
    elif abs(momentum) > 0.05:
        output.append(f"  -> Moderate trend - Watch for continuation")
    else:
        output.append(f"  -> No clear trend - Mean reversion possible")

    output.append(f"VOLATILITY: {vol*100:.1f}%")
    if vol > 0.04:
        output.append(f"  -> HIGH - Reduce position sizing by {int((1 - 0.02/vol)*100)}%")

    output.append(f"SKEWNESS: {skew:+.2f}")
    if skew < -0.5:
        output.append(f"  -> LEFT TAIL RISK - Consider put protection")
    elif skew > 0.5:
        output.append(f"  -> RIGHT SKEW - Upside potential exists")

    output.append(f"KURTOSIS: {kurt:.1f}")
    if kurt > 4:
        output.append(f"  -> FAT TAILS - VaR understates true risk")
    output.append("</QUANTITATIVE_READING>")

    # =========================================================================
    # CORRELATION INSIGHT
    # =========================================================================
    output.append("\n<CORRELATION_INSIGHT>")
    if tail_risks:
        output.append("TAIL RISK CLUSTERS DETECTED:")
        for tr in tail_risks[:3]:
            output.append(f"  {ticker} crashes with {tr['ticker']} (lambda={tr['tail']:.2f})")
        output.append("  -> Diversification ineffective in stress scenarios")
        output.append("  -> Consider index hedges instead of single-stock hedges")
    else:
        output.append("No extreme tail dependencies detected")
        output.append("Standard diversification should be effective")
    output.append("</CORRELATION_INSIGHT>")

    # =========================================================================
    # SYNTHESIS
    # =========================================================================
    output.append("\n<SYNTHESIS>")

    # Direction synthesis
    if direction == 'up':
        if momentum > 0.05:
            output.append(f"BULLISH OUTLOOK: Momentum + positive catalyst alignment")
            output.append(f"Expected move: {ret_1d:+.2f}% (1d), {ret_5d:+.2f}% (5d)")
        else:
            output.append(f"BULLISH REVERSAL: Event-driven upside against weak momentum")
    elif direction == 'down':
        if momentum < -0.05:
            output.append(f"BEARISH OUTLOOK: Momentum + negative catalyst alignment")
            output.append(f"Expected move: {ret_1d:+.2f}% (1d), {ret_5d:+.2f}% (5d)")
        else:
            output.append(f"BEARISH REVERSAL: Event-driven downside against positive momentum")
    else:
        output.append(f"NEUTRAL OUTLOOK: No strong directional signal")
        output.append(f"Expected move: {ret_1d:+.2f}% (range-bound)")

    output.append(f"CONFIDENCE: {'HIGH' if abs(ret_1d) > 2 else 'MEDIUM' if abs(ret_1d) > 1 else 'LOW'}")
    output.append("</SYNTHESIS>")

    # =========================================================================
    # OPPORTUNITIES
    # =========================================================================
    output.append("\n<OPPORTUNITIES>")

    # Momentum opportunity
    if abs(momentum) > 0.08:
        output.append(f"[MOMENTUM] {'LONG' if momentum > 0 else 'SHORT'} {ticker}")
        output.append(f"  Trend strength: {abs(momentum)*100:.1f}%")
        output.append(f"  Continue until momentum reverses")

    # Mean reversion
    if ret_1d < -3:
        output.append(f"[MEAN_REVERSION] {ticker} oversold")
        output.append(f"  Drop: {ret_1d:.1f}%")
        output.append(f"  Expected bounce: +{abs(ret_1d)*0.3:.1f}%")

    # Pairs trading
    for opp in coint_opps[:2]:
        zscore = opp['zscore'] or 0
        hedge = opp['hedge'] or 1
        if zscore < -2:
            output.append(f"[PAIRS] LONG SPREAD {ticker}/{opp['ticker']}")
            output.append(f"  Z-score: {zscore:+.2f} (mean reversion expected)")
            output.append(f"  Trade: BUY {ticker}, SELL {hedge:.2f}x {opp['ticker']}")
        elif zscore > 2:
            output.append(f"[PAIRS] SHORT SPREAD {ticker}/{opp['ticker']}")
            output.append(f"  Z-score: {zscore:+.2f} (mean reversion expected)")
            output.append(f"  Trade: SELL {ticker}, BUY {hedge:.2f}x {opp['ticker']}")

    if not coint_opps and abs(momentum) < 0.05 and abs(ret_1d) < 2:
        output.append("No strong opportunities detected")
        output.append("Wait for clearer signals")

    output.append("</OPPORTUNITIES>")

    # =========================================================================
    # RISKS
    # =========================================================================
    output.append("\n<RISKS>")

    risk_level = "LOW"
    if vix > 25 or vol > 0.04 or skew < -1 or len(tail_risks) > 2:
        risk_level = "HIGH"
    elif vix > 20 or vol > 0.03 or skew < -0.5 or len(tail_risks) > 0:
        risk_level = "ELEVATED"

    output.append(f"OVERALL_RISK: {risk_level}")

    if vix > 22:
        output.append(f"[!] VIX elevated ({vix:.1f}) - Market stress")
    if vol > 0.03:
        output.append(f"[!] High volatility ({vol*100:.1f}%) - Increase stop distance")
    if skew < -0.5:
        output.append(f"[!] Left tail risk (skew={skew:.2f}) - Put protection advised")
    if kurt > 4:
        output.append(f"[!] Fat tails (kurt={kurt:.1f}) - Black swan risk elevated")
    if tail_risks:
        output.append(f"[!] {len(tail_risks)} crash-correlated assets - Watch concentration")
    if corr > 0.5:
        output.append(f"[!] High market correlation ({corr:.2f}) - Idiosyncratic alpha reduced")

    if risk_level == "LOW":
        output.append("Standard risk management sufficient")

    output.append("</RISKS>")

    # =========================================================================
    # PORTFOLIO ACTION
    # =========================================================================
    output.append("\n<ACTION>")

    if direction == 'up' and abs(ret_1d) > 1:
        if momentum > 0.05:
            output.append(f"LONG {ticker}")
            sizing = "2-3%" if abs(ret_1d) > 2 else "1-2%"
            output.append(f"SIZING: {sizing} of portfolio")
            output.append(f"ENTRY: Current levels (momentum + catalyst)")
            output.append(f"STOP: {volatility*2:.1f}% below entry")
            output.append(f"TARGET: +{abs(ret_5d)*1.2:.1f}%")
        else:
            output.append(f"LONG {ticker} (event-driven)")
            output.append(f"SIZING: 1% of portfolio (lower conviction)")
            output.append(f"ENTRY: On pullback confirmation")
    elif direction == 'down' and abs(ret_1d) > 1:
        if momentum < -0.05:
            output.append(f"SHORT {ticker} or UNDERWEIGHT")
            output.append(f"SIZING: 2% reduction")
            output.append(f"COVER: {volatility*2:.1f}% above entry")
        elif ret_1d < -3:
            output.append(f"WATCH for bounce (mean reversion)")
            output.append(f"WAIT: 2-3 days for stabilization")
            output.append(f"POTENTIAL: +{abs(ret_1d)*0.3:.1f}%")
        else:
            output.append(f"REDUCE {ticker}")
            output.append(f"SIZING: Cut 50% of position")
    else:
        output.append(f"HOLD / NEUTRAL on {ticker}")
        output.append(f"No strong directional signal")
        output.append(f"Wait for catalyst or momentum shift")

    # Risk budget adjustment
    if risk_budget < 1:
        output.append(f"\n[RISK_BUDGET] Reduce all sizes to {risk_budget:.0%} of normal")

    output.append("</ACTION>")

    # =========================================================================
    # SAC VECTOR
    # =========================================================================
    output.append("\n<SAC_VECTOR>")
    output.append(f"regime_mode: {mode}")
    output.append(f"risk_budget: {risk_budget}")
    output.append(f"vix: {vix}")
    output.append(f"correlation: {corr}")
    output.append(f"momentum: {momentum}")
    output.append(f"volatility: {vol}")
    output.append(f"skewness: {skew}")
    output.append(f"direction_signal: {1 if direction == 'up' else -1 if direction == 'down' else 0}")
    output.append(f"expected_return_1d: {ret_1d}")
    output.append(f"expected_return_5d: {ret_5d}")
    output.append(f"confidence: {'high' if abs(ret_1d) > 2 else 'medium' if abs(ret_1d) > 1 else 'low'}")
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

        # Build text
        text = title
        if content:
            text += f". {content[:3000]}"

        # Build structured input
        structured_input = build_structured_input(
            ticker, date, text,
            daily_stats, regimes, sectors, ticker_sectors, pairwise, networks
        )

        # Build semantic output
        semantic_output = build_semantic_output(
            ticker, date, outcome,
            daily_stats, regimes, sectors, ticker_sectors, pairwise
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

        # Get date
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

        # Get market outcome
        outcome = compute_market_outcome(prices, ticker, date)
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

        # Build structured input
        structured_input = build_structured_input(
            ticker, date, text,
            daily_stats, regimes, sectors, ticker_sectors, pairwise, networks
        )

        # Build semantic output
        semantic_output = build_semantic_output(
            ticker, date, outcome,
            daily_stats, regimes, sectors, ticker_sectors, pairwise
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
    parser = argparse.ArgumentParser(description="Prepare SOTA training data for Promethee")
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
    logger.info("PROMETHEE - SOTA Semantic Training Data Preparation")
    logger.info("=" * 70)
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Max news: {args.max_news:,}, Max SEC: {args.max_sec:,}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info("")

    # Connect
    conn = sqlite3.connect(args.db_path)

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
        pairwise, networks,
        args.start_date, args.end_date, args.max_news, args.news_sample_rate
    )

    # Process SEC
    sec_samples = process_sec_filings(
        conn, prices, daily_stats, regimes, sectors, ticker_sectors,
        pairwise, networks, args.max_sec
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
        logger.info("--- INPUT (first 500 chars) ---")
        logger.info(sample['input'][:500])
        logger.info("")
        logger.info("--- OUTPUT (first 500 chars) ---")
        logger.info(sample['output'][:500])


if __name__ == "__main__":
    main()
