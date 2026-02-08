"""
Geometric Alpha Analysis on S&P 500
====================================

This script analyzes whether geometric features contain alpha signals:
1. Compute TDA, Ricci Curvature, Takens, Fisher-Rao on historical data
2. Test predictive power for future returns
3. Build simple trading signals based on geometry
4. Backtest and measure alpha

Usage:
    python scripts/geometric_alpha_analysis.py --db_path data/eodhd_sp500.db
"""

import argparse
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Geometric libraries
try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("Warning: ripser not installed. Using simplified TDA.")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed.")

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from scipy.stats import entropy


# =============================================================================
# Geometric Computers (simplified versions for analysis)
# =============================================================================

class TDAComputer:
    """Compute TDA features from price series."""

    def __init__(self, window_size: int = 20):
        self.window_size = window_size

    def sliding_window_embedding(self, series: np.ndarray) -> Optional[np.ndarray]:
        """Create point cloud from time series."""
        n = len(series)
        if n < self.window_size:
            return None

        points = []
        for i in range(n - self.window_size + 1):
            points.append(series[i:i + self.window_size])

        return np.array(points)

    def compute_features(self, series: np.ndarray) -> dict:
        """Compute TDA features."""
        points = self.sliding_window_embedding(series)

        if points is None or len(points) < 10:
            return {'betti_0': 0, 'betti_1': 0, 'persistence_entropy': 0, 'total_persistence': 0}

        # Normalize
        points = (points - points.mean()) / (points.std() + 1e-8)

        if HAS_RIPSER:
            try:
                result = ripser(points, maxdim=1, thresh=2.0)
                diagrams = result['dgms']

                # Betti numbers
                betti_0 = len(diagrams[0])
                betti_1 = len([p for p in diagrams[1] if np.isfinite(p[1])])

                # Persistence
                all_lifetimes = []
                for dim in range(2):
                    diagram = diagrams[dim]
                    finite_mask = np.isfinite(diagram[:, 1])
                    diagram = diagram[finite_mask]
                    if len(diagram) > 0:
                        lifetimes = diagram[:, 1] - diagram[:, 0]
                        all_lifetimes.extend(lifetimes[lifetimes > 0])

                if all_lifetimes:
                    all_lifetimes = np.array(all_lifetimes)
                    probs = all_lifetimes / all_lifetimes.sum()
                    pers_entropy = -np.sum(probs * np.log(probs + 1e-10))
                    total_pers = np.sum(all_lifetimes)
                else:
                    pers_entropy = 0
                    total_pers = 0

                return {
                    'betti_0': betti_0,
                    'betti_1': betti_1,
                    'persistence_entropy': pers_entropy,
                    'total_persistence': total_pers
                }
            except:
                pass

        # Fallback: distance-based approximation
        dists = pdist(points)
        return {
            'betti_0': len(points),
            'betti_1': 0,
            'persistence_entropy': entropy(np.histogram(dists, bins=20)[0] + 1),
            'total_persistence': np.sum(dists)
        }


class RicciCurvatureComputer:
    """Compute Ricci curvature on correlation network."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold

    def compute_curvatures(self, corr_matrix: np.ndarray, tickers: List[str]) -> Dict[str, float]:
        """Compute curvature for each ticker."""
        if not HAS_NETWORKX:
            return {t: 0 for t in tickers}

        G = nx.Graph()
        n = len(tickers)

        for i in range(n):
            G.add_node(tickers[i])

        for i in range(n):
            for j in range(i + 1, n):
                corr = corr_matrix[i, j]
                if abs(corr) > self.threshold:
                    G.add_edge(tickers[i], tickers[j], weight=1 - abs(corr))

        curvatures = {}
        for node in tickers:
            if node in G:
                # Use clustering coefficient as curvature proxy
                clustering = nx.clustering(G, node)
                curvatures[node] = 2 * clustering - 1  # Map to [-1, 1]
            else:
                curvatures[node] = 0

        return curvatures

    def compute_network_curvature(self, corr_matrix: np.ndarray, tickers: List[str]) -> float:
        """Compute average network curvature."""
        curvatures = self.compute_curvatures(corr_matrix, tickers)
        if curvatures:
            return np.mean(list(curvatures.values()))
        return 0


class TakensEmbedding:
    """Phase space reconstruction."""

    def __init__(self, embedding_dim: int = 10, tau: int = 5):
        self.embedding_dim = embedding_dim
        self.tau = tau

    def compute_features(self, series: np.ndarray) -> dict:
        """Compute phase space features."""
        n = len(series)
        required_length = (self.embedding_dim - 1) * self.tau + 1

        if n < required_length:
            return {'lyapunov_proxy': 0, 'recurrence_rate': 0, 'determinism': 0}

        # Normalize
        series = (series - np.mean(series)) / (np.std(series) + 1e-8)

        # Create embedding
        embedded = []
        for i in range(n - (self.embedding_dim - 1) * self.tau):
            point = [series[i + j * self.tau] for j in range(self.embedding_dim)]
            embedded.append(point)

        embedded = np.array(embedded)
        if len(embedded) < 10:
            return {'lyapunov_proxy': 0, 'recurrence_rate': 0, 'determinism': 0}

        # Recurrence rate
        dists = squareform(pdist(embedded))
        threshold = np.percentile(dists, 10)
        recurrence_matrix = dists < threshold
        recurrence_rate = np.sum(recurrence_matrix) / (len(embedded) ** 2)

        # Determinism (autocorrelation proxy)
        autocorr = np.correlate(embedded[:, 0], embedded[:, 0], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-8)
        determinism = np.mean(autocorr[:min(10, len(autocorr))])

        # Lyapunov proxy
        lyapunov_proxy = 0
        if len(embedded) > 20:
            n_samples = min(30, len(embedded) - 10)
            divergences = []
            for i in range(n_samples):
                d = dists[i].copy()
                d[i] = np.inf
                nearest = np.argmin(d)

                if i + 5 < len(embedded) and nearest + 5 < len(embedded):
                    initial_dist = dists[i, nearest]
                    final_dist = np.linalg.norm(embedded[i + 5] - embedded[nearest + 5])
                    if initial_dist > 1e-8:
                        divergences.append(np.log(final_dist / initial_dist + 1e-8) / 5)

            if divergences:
                lyapunov_proxy = np.mean(divergences)

        return {
            'lyapunov_proxy': lyapunov_proxy,
            'recurrence_rate': recurrence_rate,
            'determinism': determinism
        }


class FisherRaoComputer:
    """Fisher-Rao distance computation."""

    def compute_distance(self, returns1: np.ndarray, returns2: np.ndarray) -> float:
        """Compute Fisher-Rao (Hellinger) distance between return distributions."""
        mu1, sigma1 = np.mean(returns1), np.std(returns1) + 1e-8
        mu2, sigma2 = np.mean(returns2), np.std(returns2) + 1e-8

        term1 = (sigma1 * sigma2) / (sigma1**2 + sigma2**2)
        term2 = np.exp(-0.25 * (mu1 - mu2)**2 / (sigma1**2 + sigma2**2))

        hellinger = np.sqrt(1 - np.sqrt(2 * term1) * term2)
        return hellinger


# =============================================================================
# Data Loading
# =============================================================================

def load_sp500_data(db_path: str, start_date: str = "2015-01-01") -> pd.DataFrame:
    """Load S&P 500 price data."""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT DATE(date) as date, ticker, adjusted_close
        FROM historical_prices
        WHERE DATE(date) >= ?
        ORDER BY date, ticker
    """

    df = pd.read_sql_query(query, conn, params=(start_date,))
    conn.close()

    print(f"Loaded {len(df)} rows, {df['ticker'].nunique()} unique tickers")

    # Remove non-stock tickers (like EURUSD=X)
    df = df[~df['ticker'].str.contains('=', na=False)]
    df = df[df['adjusted_close'] > 0]

    print(f"After filtering: {len(df)} rows, {df['ticker'].nunique()} tickers")

    # Pivot to price matrix
    price_matrix = df.pivot(index='date', columns='ticker', values='adjusted_close')
    price_matrix = price_matrix.sort_index()

    # Keep tickers with enough data (at least 80% of dates)
    min_obs = len(price_matrix) * 0.8
    valid_tickers = price_matrix.columns[price_matrix.notna().sum() > min_obs]
    price_matrix = price_matrix[valid_tickers]

    print(f"After coverage filter: {len(price_matrix)} dates, {len(price_matrix.columns)} tickers")

    # Forward fill missing values
    price_matrix = price_matrix.ffill()

    return price_matrix


# =============================================================================
# Geometric Feature Computation
# =============================================================================

def compute_geometric_features(
    price_matrix: pd.DataFrame,
    window: int = 60,
    step: int = 5
) -> pd.DataFrame:
    """Compute all geometric features for each date."""

    returns_matrix = price_matrix.pct_change().dropna()
    dates = returns_matrix.index.tolist()
    tickers = returns_matrix.columns.tolist()

    # Initialize computers
    tda = TDAComputer(window_size=20)
    ricci = RicciCurvatureComputer(threshold=0.3)
    takens = TakensEmbedding(embedding_dim=10, tau=5)
    fisher = FisherRaoComputer()

    results = []

    print(f"Computing geometric features for {len(dates) - window} dates...")

    for i in tqdm(range(window, len(dates), step)):
        current_date = dates[i]
        window_returns = returns_matrix.iloc[i-window:i]
        window_prices = price_matrix.iloc[i-window:i]

        # Valid tickers
        valid = window_returns.columns[window_returns.isna().sum() < window * 0.2].tolist()
        if len(valid) < 10:
            continue

        window_returns = window_returns[valid].fillna(0)
        window_prices = window_prices[valid].ffill()

        # 1. Market-level TDA
        market_returns = window_returns.mean(axis=1).values
        tda_features = tda.compute_features(market_returns)

        # 2. Correlation network Ricci curvature
        corr_matrix = window_returns.corr().values
        avg_curvature = ricci.compute_network_curvature(corr_matrix, valid)

        # 3. Market-level Takens
        market_prices = window_prices.mean(axis=1).values
        takens_features = takens.compute_features(market_prices)

        # 4. Dispersion (Fisher-Rao aggregate)
        all_returns = window_returns.values.flatten()
        market_mu = np.mean(all_returns)
        market_sigma = np.std(all_returns)

        dispersion = 0
        for ticker in valid[:50]:  # Sample for speed
            ticker_returns = window_returns[ticker].values
            d = fisher.compute_distance(ticker_returns, all_returns)
            dispersion += d
        dispersion /= min(len(valid), 50)

        # 5. Future returns (1d, 5d, 20d)
        future_1d = None
        future_5d = None
        future_20d = None

        if i + 1 < len(dates):
            future_1d = returns_matrix.iloc[i+1].mean() * 100
        if i + 5 < len(dates):
            future_5d = returns_matrix.iloc[i+1:i+6].mean().mean() * 100
        if i + 20 < len(dates):
            future_20d = returns_matrix.iloc[i+1:i+21].mean().mean() * 100

        results.append({
            'date': current_date,
            # TDA
            'betti_0': tda_features['betti_0'],
            'betti_1': tda_features['betti_1'],
            'persistence_entropy': tda_features['persistence_entropy'],
            'total_persistence': tda_features['total_persistence'],
            # Ricci
            'network_curvature': avg_curvature,
            # Takens
            'lyapunov_proxy': takens_features['lyapunov_proxy'],
            'recurrence_rate': takens_features['recurrence_rate'],
            'determinism': takens_features['determinism'],
            # Fisher-Rao
            'fisher_dispersion': dispersion,
            # Market stats
            'market_volatility': market_sigma * np.sqrt(252) * 100,
            'market_correlation': np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),
            # Future returns
            'future_1d': future_1d,
            'future_5d': future_5d,
            'future_20d': future_20d
        })

    return pd.DataFrame(results)


# =============================================================================
# Alpha Analysis
# =============================================================================

def analyze_predictive_power(df: pd.DataFrame):
    """Analyze correlation between geometric features and future returns."""

    print("\n" + "="*70)
    print("GEOMETRIC FEATURES vs FUTURE RETURNS")
    print("="*70)

    if len(df) == 0:
        print("No data to analyze!")
        return pd.DataFrame()

    features = [f for f in [
        'betti_1', 'persistence_entropy', 'total_persistence',
        'network_curvature', 'lyapunov_proxy', 'recurrence_rate',
        'determinism', 'fisher_dispersion', 'market_volatility', 'market_correlation'
    ] if f in df.columns]

    horizons = ['future_1d', 'future_5d', 'future_20d']

    results = []

    for feature in features:
        for horizon in horizons:
            # Remove NaN
            mask = df[feature].notna() & df[horizon].notna()
            x = df.loc[mask, feature].values
            y = df.loc[mask, horizon].values

            if len(x) < 30:
                continue

            # Pearson correlation
            pearson_r, pearson_p = pearsonr(x, y)

            # Spearman correlation (rank-based, more robust)
            spearman_r, spearman_p = spearmanr(x, y)

            results.append({
                'feature': feature,
                'horizon': horizon,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'significant': spearman_p < 0.05
            })

    results_df = pd.DataFrame(results)

    # Print significant results
    print("\nSignificant correlations (p < 0.05):")
    print("-" * 70)

    sig_results = results_df[results_df['significant']].sort_values('spearman_p')

    if len(sig_results) > 0:
        for _, row in sig_results.iterrows():
            direction = "+" if row['spearman_r'] > 0 else "-"
            print(f"{row['feature']:25s} -> {row['horizon']:12s}: "
                  f"r={row['spearman_r']:+.3f} (p={row['spearman_p']:.4f}) {direction}")
    else:
        print("No significant correlations found at p < 0.05")

    print("\n" + "-" * 70)
    print("All correlations:")
    print(results_df.pivot_table(
        index='feature',
        columns='horizon',
        values='spearman_r'
    ).round(3).to_string())

    return results_df


def build_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Build trading signals from geometric features."""

    print("\n" + "="*70)
    print("GEOMETRIC TRADING SIGNALS")
    print("="*70)

    df = df.copy()

    # Signal 1: Low Curvature = Systemic Risk = Reduce exposure
    # When network curvature is very negative, market is fragile
    df['signal_curvature'] = np.where(
        df['network_curvature'] < df['network_curvature'].rolling(20).mean() - df['network_curvature'].rolling(20).std(),
        -1,  # Reduce exposure
        np.where(
            df['network_curvature'] > df['network_curvature'].rolling(20).mean() + df['network_curvature'].rolling(20).std(),
            1,  # Increase exposure
            0   # Neutral
        )
    )

    # Signal 2: High Lyapunov = Chaos = Reduce position size
    df['signal_chaos'] = np.where(
        df['lyapunov_proxy'] > df['lyapunov_proxy'].quantile(0.8),
        -1,  # High chaos, reduce
        np.where(
            df['lyapunov_proxy'] < df['lyapunov_proxy'].quantile(0.2),
            1,  # Low chaos, stable
            0
        )
    )

    # Signal 3: High Persistence Entropy = Regime instability
    df['signal_topology'] = np.where(
        df['persistence_entropy'] > df['persistence_entropy'].quantile(0.8),
        -1,  # Unstable topology
        np.where(
            df['persistence_entropy'] < df['persistence_entropy'].quantile(0.2),
            1,  # Stable topology
            0
        )
    )

    # Signal 4: High dispersion = Stock-picking opportunity
    df['signal_dispersion'] = np.where(
        df['fisher_dispersion'] > df['fisher_dispersion'].quantile(0.7),
        1,  # High dispersion, good for stock picking
        np.where(
            df['fisher_dispersion'] < df['fisher_dispersion'].quantile(0.3),
            -1,  # Low dispersion, beta-dominated
            0
        )
    )

    # Signal 5: High Recurrence = Pattern repeating
    df['signal_recurrence'] = np.where(
        df['recurrence_rate'] > df['recurrence_rate'].quantile(0.8),
        1,  # Pattern repeating, trend continuation
        0
    )

    # Composite signal
    df['composite_signal'] = (
        df['signal_curvature'] * 0.25 +
        df['signal_chaos'] * 0.25 +
        df['signal_topology'] * 0.2 +
        df['signal_dispersion'] * 0.15 +
        df['signal_recurrence'] * 0.15
    )

    return df


def backtest_signals(df: pd.DataFrame):
    """Backtest geometric signals."""

    print("\n" + "="*70)
    print("BACKTEST RESULTS")
    print("="*70)

    signals = ['signal_curvature', 'signal_chaos', 'signal_topology',
               'signal_dispersion', 'signal_recurrence', 'composite_signal']

    results = []

    for signal in signals:
        if signal not in df.columns:
            continue

        # Calculate returns by signal quintile
        df_clean = df[[signal, 'future_5d']].dropna()

        if len(df_clean) < 50:
            continue

        # Long when signal > 0, short when < 0
        long_mask = df_clean[signal] > 0
        short_mask = df_clean[signal] < 0
        neutral_mask = df_clean[signal] == 0

        long_returns = df_clean.loc[long_mask, 'future_5d'].mean() if long_mask.sum() > 0 else 0
        short_returns = df_clean.loc[short_mask, 'future_5d'].mean() if short_mask.sum() > 0 else 0
        neutral_returns = df_clean.loc[neutral_mask, 'future_5d'].mean() if neutral_mask.sum() > 0 else 0

        # Long-short spread
        ls_spread = long_returns - short_returns

        # Information coefficient (correlation with future returns)
        ic, ic_p = spearmanr(df_clean[signal], df_clean['future_5d'])

        results.append({
            'signal': signal,
            'long_return': long_returns,
            'short_return': short_returns,
            'neutral_return': neutral_returns,
            'long_short_spread': ls_spread,
            'ic': ic,
            'ic_pvalue': ic_p,
            'n_long': long_mask.sum(),
            'n_short': short_mask.sum(),
            'n_neutral': neutral_mask.sum()
        })

    results_df = pd.DataFrame(results)

    print("\nSignal Performance (5-day forward returns):")
    print("-" * 70)
    print(f"{'Signal':<25} {'Long':>8} {'Short':>8} {'L-S':>8} {'IC':>8} {'p-val':>8}")
    print("-" * 70)

    for _, row in results_df.iterrows():
        sig = "*" if row['ic_pvalue'] < 0.05 else ""
        print(f"{row['signal']:<25} {row['long_return']:>7.3f}% {row['short_return']:>7.3f}% "
              f"{row['long_short_spread']:>7.3f}% {row['ic']:>7.3f} {row['ic_pvalue']:>7.4f}{sig}")

    print("\n* = statistically significant (p < 0.05)")

    return results_df


def create_visualizations(df: pd.DataFrame, output_dir: str = "."):
    """Create visualizations of geometric features."""

    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # 1. Network Curvature over time
    ax = axes[0, 0]
    ax.plot(pd.to_datetime(df['date']), df['network_curvature'], 'b-', alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Network Ricci Curvature')
    ax.set_ylabel('Curvature')
    ax.fill_between(pd.to_datetime(df['date']), df['network_curvature'], 0,
                    where=df['network_curvature'] < 0, color='red', alpha=0.3, label='Negative (Fragile)')
    ax.fill_between(pd.to_datetime(df['date']), df['network_curvature'], 0,
                    where=df['network_curvature'] > 0, color='green', alpha=0.3, label='Positive (Stable)')
    ax.legend(loc='upper right', fontsize=8)

    # 2. Persistence Entropy
    ax = axes[0, 1]
    ax.plot(pd.to_datetime(df['date']), df['persistence_entropy'], 'purple', alpha=0.7)
    ax.set_title('TDA Persistence Entropy')
    ax.set_ylabel('Entropy')

    # 3. Lyapunov Proxy
    ax = axes[0, 2]
    ax.plot(pd.to_datetime(df['date']), df['lyapunov_proxy'], 'orange', alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Lyapunov Proxy (Chaos)')
    ax.set_ylabel('Lyapunov')

    # 4. Fisher-Rao Dispersion
    ax = axes[1, 0]
    ax.plot(pd.to_datetime(df['date']), df['fisher_dispersion'], 'green', alpha=0.7)
    ax.set_title('Fisher-Rao Dispersion')
    ax.set_ylabel('Dispersion')

    # 5. Curvature vs Future Returns
    ax = axes[1, 1]
    mask = df['network_curvature'].notna() & df['future_5d'].notna()
    ax.scatter(df.loc[mask, 'network_curvature'], df.loc[mask, 'future_5d'],
               alpha=0.3, s=10)
    ax.set_xlabel('Network Curvature')
    ax.set_ylabel('Future 5d Return (%)')
    ax.set_title('Curvature vs Future Returns')
    # Add trend line
    z = np.polyfit(df.loc[mask, 'network_curvature'], df.loc[mask, 'future_5d'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df.loc[mask, 'network_curvature'].min(), df.loc[mask, 'network_curvature'].max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
    ax.legend()

    # 6. Lyapunov vs Future Returns
    ax = axes[1, 2]
    mask = df['lyapunov_proxy'].notna() & df['future_5d'].notna()
    ax.scatter(df.loc[mask, 'lyapunov_proxy'], df.loc[mask, 'future_5d'],
               alpha=0.3, s=10, c='orange')
    ax.set_xlabel('Lyapunov Proxy')
    ax.set_ylabel('Future 5d Return (%)')
    ax.set_title('Chaos vs Future Returns')

    # 7. Betti numbers over time
    ax = axes[2, 0]
    ax.plot(pd.to_datetime(df['date']), df['betti_1'], 'purple', alpha=0.7, label='Betti-1 (Cycles)')
    ax.set_title('TDA Betti Numbers')
    ax.set_ylabel('Betti-1')
    ax.legend()

    # 8. Composite signal
    if 'composite_signal' in df.columns:
        ax = axes[2, 1]
        ax.plot(pd.to_datetime(df['date']), df['composite_signal'], 'blue', alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_title('Composite Geometric Signal')
        ax.set_ylabel('Signal')
        ax.fill_between(pd.to_datetime(df['date']), df['composite_signal'], 0,
                        where=df['composite_signal'] > 0, color='green', alpha=0.3)
        ax.fill_between(pd.to_datetime(df['date']), df['composite_signal'], 0,
                        where=df['composite_signal'] < 0, color='red', alpha=0.3)

    # 9. Cumulative strategy returns
    ax = axes[2, 2]
    if 'composite_signal' in df.columns:
        df_clean = df[['composite_signal', 'future_5d']].dropna()
        df_clean['strategy_return'] = df_clean['composite_signal'].shift(1) * df_clean['future_5d']
        df_clean['cumulative'] = (1 + df_clean['strategy_return']/100).cumprod()
        df_clean['buy_hold'] = (1 + df_clean['future_5d']/100).cumprod()

        ax.plot(df_clean['cumulative'].values, 'b-', label='Geometric Strategy', linewidth=2)
        ax.plot(df_clean['buy_hold'].values, 'gray', label='Buy & Hold', alpha=0.5)
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Growth of $1')
        ax.legend()

    plt.tight_layout()

    output_path = f"{output_dir}/geometric_alpha_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")

    plt.show()


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics of geometric features."""

    print("\n" + "="*70)
    print("GEOMETRIC FEATURE STATISTICS")
    print("="*70)

    features = [
        'betti_0', 'betti_1', 'persistence_entropy', 'total_persistence',
        'network_curvature', 'lyapunov_proxy', 'recurrence_rate',
        'determinism', 'fisher_dispersion'
    ]

    print(f"\n{'Feature':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 70)

    for feat in features:
        if feat in df.columns:
            print(f"{feat:<25} {df[feat].mean():>10.4f} {df[feat].std():>10.4f} "
                  f"{df[feat].min():>10.4f} {df[feat].max():>10.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Geometric Alpha Analysis on S&P 500")
    parser.add_argument("--db_path", type=str, required=True, help="Path to database")
    parser.add_argument("--start_date", type=str, default="2015-01-01", help="Start date")
    parser.add_argument("--window", type=int, default=60, help="Rolling window size")
    parser.add_argument("--step", type=int, default=5, help="Step size in days")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    print("="*70)
    print("GEOMETRIC ALPHA ANALYSIS - S&P 500")
    print("="*70)
    print(f"Database: {args.db_path}")
    print(f"Start date: {args.start_date}")
    print(f"Window: {args.window} days")
    print()

    # 1. Load data
    print("Loading S&P 500 data...")
    price_matrix = load_sp500_data(args.db_path, args.start_date)
    print(f"Loaded {len(price_matrix)} dates, {len(price_matrix.columns)} tickers")

    # 2. Compute geometric features
    df = compute_geometric_features(price_matrix, window=args.window, step=args.step)
    print(f"\nComputed features for {len(df)} data points")

    # 3. Print statistics
    print_summary_statistics(df)

    # 4. Analyze predictive power
    correlations = analyze_predictive_power(df)

    # 5. Build trading signals
    df = build_trading_signals(df)

    # 6. Backtest
    backtest_results = backtest_signals(df)

    # 7. Visualizations
    create_visualizations(df, args.output_dir)

    # 8. Save results
    output_csv = f"{args.output_dir}/geometric_features.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved features to: {output_csv}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

    # Key findings
    print("\nKEY FINDINGS:")
    print("-" * 70)

    # Check for significant predictors
    sig_predictors = correlations[correlations['significant']]
    if len(sig_predictors) > 0:
        print(f"Found {len(sig_predictors)} statistically significant geometric predictors:")
        for _, row in sig_predictors.head(5).iterrows():
            print(f"  - {row['feature']} -> {row['horizon']}: r={row['spearman_r']:.3f}")
    else:
        print("No statistically significant predictors found at p < 0.05")
        print("However, some features may still have economic significance:")
        top_corr = correlations.nlargest(3, 'spearman_r')
        for _, row in top_corr.iterrows():
            print(f"  - {row['feature']} -> {row['horizon']}: r={row['spearman_r']:.3f} (p={row['spearman_p']:.3f})")

    # Best signals
    if len(backtest_results) > 0:
        best_signal = backtest_results.loc[backtest_results['long_short_spread'].abs().idxmax()]
        print(f"\nBest trading signal: {best_signal['signal']}")
        print(f"  Long-Short spread: {best_signal['long_short_spread']:.3f}% per 5 days")
        print(f"  Information Coefficient: {best_signal['ic']:.3f}")


if __name__ == "__main__":
    main()
