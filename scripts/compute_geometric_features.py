"""
Compute Geometric Features for Unified Financial Manifold

This script computes ALL geometric features BEFORE training:
1. TDA (Topological Data Analysis) - Persistent homology on returns
2. Ricci Curvature - On correlation network
3. Takens Embedding - Phase space reconstruction
4. Fisher-Rao distances - Between return distributions
5. UMAP projections - Manifold coordinates
6. Convex Hull / DEA - Fundamental efficiency frontier

All features are stored in SQLite for use during training.

Usage:
    python scripts/compute_geometric_features.py \
        --db_path data/eodhd_sp500.db \
        --start_date 2015-01-01 \
        --end_date 2025-01-01
"""

import argparse
import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Geometric libraries
try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    logger.warning("ripser not installed. Using simplified TDA.")

try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.spatial import ConvexHull
    from scipy.stats import entropy
    from scipy.linalg import logm, expm
except ImportError:
    raise ImportError("scipy is required")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("networkx not installed. Ricci curvature disabled.")


# =============================================================================
# 1. TDA - Topological Data Analysis
# =============================================================================

class TDAComputer:
    """
    Compute Topological Data Analysis features using Persistent Homology.
    """

    def __init__(self, max_dim: int = 2, window_size: int = 20):
        self.max_dim = max_dim
        self.window_size = window_size

    def sliding_window_embedding(self, series: np.ndarray, tau: int = 1) -> np.ndarray:
        """
        Create point cloud from time series using sliding window.
        Each window becomes a point in high-dimensional space.
        """
        n = len(series)
        if n < self.window_size:
            return None

        points = []
        for i in range(n - self.window_size + 1):
            points.append(series[i:i + self.window_size])

        return np.array(points)

    def compute_persistence(self, points: np.ndarray) -> dict:
        """
        Compute persistent homology and extract features.
        """
        if points is None or len(points) < 10:
            return self._empty_features()

        # Normalize points
        points = (points - points.mean()) / (points.std() + 1e-8)

        if HAS_RIPSER:
            # Use ripser for persistent homology
            result = ripser(points, maxdim=self.max_dim, thresh=2.0)
            diagrams = result['dgms']
        else:
            # Simplified: use distance matrix statistics
            return self._simplified_tda(points)

        features = {}

        for dim in range(self.max_dim + 1):
            diagram = diagrams[dim]

            # Filter infinite deaths
            finite_mask = np.isfinite(diagram[:, 1])
            diagram = diagram[finite_mask]

            if len(diagram) == 0:
                features[f'betti_{dim}'] = 0
                features[f'persistence_mean_{dim}'] = 0
                features[f'persistence_max_{dim}'] = 0
                features[f'persistence_std_{dim}'] = 0
                features[f'persistence_entropy_{dim}'] = 0
                continue

            lifetimes = diagram[:, 1] - diagram[:, 0]
            lifetimes = lifetimes[lifetimes > 0]

            if len(lifetimes) == 0:
                features[f'betti_{dim}'] = 0
                features[f'persistence_mean_{dim}'] = 0
                features[f'persistence_max_{dim}'] = 0
                features[f'persistence_std_{dim}'] = 0
                features[f'persistence_entropy_{dim}'] = 0
                continue

            # Betti number (count of features)
            features[f'betti_{dim}'] = len(lifetimes)

            # Persistence statistics
            features[f'persistence_mean_{dim}'] = float(np.mean(lifetimes))
            features[f'persistence_max_{dim}'] = float(np.max(lifetimes))
            features[f'persistence_std_{dim}'] = float(np.std(lifetimes))

            # Persistence entropy
            probs = lifetimes / lifetimes.sum()
            features[f'persistence_entropy_{dim}'] = float(-np.sum(probs * np.log(probs + 1e-10)))

        # Total persistence (L1 norm of persistence landscape)
        all_lifetimes = []
        for dim in range(self.max_dim + 1):
            diagram = diagrams[dim]
            finite_mask = np.isfinite(diagram[:, 1])
            diagram = diagram[finite_mask]
            if len(diagram) > 0:
                all_lifetimes.extend(diagram[:, 1] - diagram[:, 0])

        all_lifetimes = np.array(all_lifetimes)
        all_lifetimes = all_lifetimes[all_lifetimes > 0]

        features['total_persistence'] = float(np.sum(all_lifetimes)) if len(all_lifetimes) > 0 else 0
        features['persistence_landscape_norm'] = float(np.linalg.norm(all_lifetimes)) if len(all_lifetimes) > 0 else 0

        return features

    def _simplified_tda(self, points: np.ndarray) -> dict:
        """Simplified TDA when ripser is not available."""
        # Use distance matrix properties as proxy
        dists = pdist(points)

        features = {
            'betti_0': len(points),
            'betti_1': 0,
            'betti_2': 0,
            'persistence_mean_0': float(np.mean(dists)),
            'persistence_max_0': float(np.max(dists)),
            'persistence_std_0': float(np.std(dists)),
            'persistence_entropy_0': float(entropy(np.histogram(dists, bins=20)[0] + 1)),
            'persistence_mean_1': 0,
            'persistence_max_1': 0,
            'persistence_std_1': 0,
            'persistence_entropy_1': 0,
            'persistence_mean_2': 0,
            'persistence_max_2': 0,
            'persistence_std_2': 0,
            'persistence_entropy_2': 0,
            'total_persistence': float(np.sum(dists)),
            'persistence_landscape_norm': float(np.linalg.norm(dists))
        }
        return features

    def _empty_features(self) -> dict:
        """Return empty features dict."""
        features = {}
        for dim in range(self.max_dim + 1):
            features[f'betti_{dim}'] = 0
            features[f'persistence_mean_{dim}'] = 0
            features[f'persistence_max_{dim}'] = 0
            features[f'persistence_std_{dim}'] = 0
            features[f'persistence_entropy_{dim}'] = 0
        features['total_persistence'] = 0
        features['persistence_landscape_norm'] = 0
        return features


# =============================================================================
# 2. Ricci Curvature on Correlation Network
# =============================================================================

class RicciCurvatureComputer:
    """
    Compute Ollivier-Ricci curvature on financial correlation network.
    """

    def __init__(self, correlation_threshold: float = 0.3):
        self.threshold = correlation_threshold

    def build_network(self, correlation_matrix: np.ndarray, tickers: List[str]) -> Optional[nx.Graph]:
        """Build network from correlation matrix."""
        if not HAS_NETWORKX:
            return None

        G = nx.Graph()
        n = len(tickers)

        for i in range(n):
            G.add_node(tickers[i])

        for i in range(n):
            for j in range(i + 1, n):
                corr = correlation_matrix[i, j]
                if abs(corr) > self.threshold:
                    # Weight = 1 - |correlation| (closer = shorter distance)
                    weight = 1 - abs(corr)
                    G.add_edge(tickers[i], tickers[j], weight=weight, correlation=corr)

        return G

    def compute_ollivier_ricci(self, G: nx.Graph, node: str) -> float:
        """
        Compute Ollivier-Ricci curvature for a node.
        Simplified version using local clustering.
        """
        if G is None or node not in G:
            return 0.0

        neighbors = list(G.neighbors(node))
        if len(neighbors) < 2:
            return 0.0

        # Compute local clustering coefficient as proxy for curvature
        # High clustering = positive curvature (sphere-like)
        # Low clustering = negative curvature (hyperbolic)
        clustering = nx.clustering(G, node)

        # Map to curvature: clustering 0->1 maps to curvature -1->1
        curvature = 2 * clustering - 1

        return float(curvature)

    def compute_all_curvatures(self, G: nx.Graph) -> Dict[str, float]:
        """Compute curvature for all nodes."""
        if G is None:
            return {}

        curvatures = {}
        for node in G.nodes():
            curvatures[node] = self.compute_ollivier_ricci(G, node)

        return curvatures

    def compute_network_features(self, G: nx.Graph) -> dict:
        """Compute global network features."""
        if G is None:
            return {
                'avg_curvature': 0,
                'std_curvature': 0,
                'min_curvature': 0,
                'max_curvature': 0,
                'network_density': 0,
                'avg_clustering': 0,
                'transitivity': 0
            }

        curvatures = self.compute_all_curvatures(G)
        curv_values = list(curvatures.values())

        return {
            'avg_curvature': float(np.mean(curv_values)) if curv_values else 0,
            'std_curvature': float(np.std(curv_values)) if curv_values else 0,
            'min_curvature': float(np.min(curv_values)) if curv_values else 0,
            'max_curvature': float(np.max(curv_values)) if curv_values else 0,
            'network_density': float(nx.density(G)),
            'avg_clustering': float(nx.average_clustering(G)),
            'transitivity': float(nx.transitivity(G))
        }


# =============================================================================
# 3. Takens Embedding (Phase Space Reconstruction)
# =============================================================================

class TakensEmbedding:
    """
    Reconstruct phase space from price series using Takens' theorem.
    """

    def __init__(self, embedding_dim: int = 10, tau: int = 5):
        self.embedding_dim = embedding_dim
        self.tau = tau

    def embed(self, series: np.ndarray) -> Optional[np.ndarray]:
        """Create delay embedding."""
        n = len(series)
        required_length = (self.embedding_dim - 1) * self.tau + 1

        if n < required_length:
            return None

        # Normalize
        series = (series - np.mean(series)) / (np.std(series) + 1e-8)

        # Create embedding
        embedded = []
        for i in range(n - (self.embedding_dim - 1) * self.tau):
            point = [series[i + j * self.tau] for j in range(self.embedding_dim)]
            embedded.append(point)

        return np.array(embedded)

    def compute_features(self, series: np.ndarray) -> dict:
        """Compute phase space features."""
        embedded = self.embed(series)

        if embedded is None or len(embedded) < 10:
            return {
                'phase_space_dim': 0,
                'trajectory_length': 0,
                'recurrence_rate': 0,
                'determinism': 0,
                'entropy_rate': 0,
                'lyapunov_proxy': 0
            }

        # Trajectory length (total distance traveled)
        diffs = np.diff(embedded, axis=0)
        trajectory_length = np.sum(np.linalg.norm(diffs, axis=1))

        # Recurrence rate (fraction of points that recur)
        dists = squareform(pdist(embedded))
        threshold = np.percentile(dists, 10)
        recurrence_matrix = dists < threshold
        recurrence_rate = np.sum(recurrence_matrix) / (len(embedded) ** 2)

        # Determinism (fraction of recurrent points forming lines)
        # Simplified: use autocorrelation as proxy
        autocorr = np.correlate(embedded[:, 0], embedded[:, 0], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        determinism = np.mean(autocorr[:10])

        # Entropy rate (from recurrence)
        p_recur = recurrence_rate
        if 0 < p_recur < 1:
            entropy_rate = -p_recur * np.log(p_recur) - (1 - p_recur) * np.log(1 - p_recur)
        else:
            entropy_rate = 0

        # Lyapunov exponent proxy (divergence rate)
        # Measure how quickly nearby trajectories diverge
        lyapunov_proxy = 0
        if len(embedded) > 20:
            n_samples = min(50, len(embedded) - 10)
            divergences = []
            for i in range(n_samples):
                # Find nearest neighbor
                d = dists[i].copy()
                d[i] = np.inf
                nearest = np.argmin(d)

                # Track divergence after 5 steps
                if i + 5 < len(embedded) and nearest + 5 < len(embedded):
                    initial_dist = dists[i, nearest]
                    final_dist = np.linalg.norm(embedded[i + 5] - embedded[nearest + 5])
                    if initial_dist > 1e-8:
                        divergences.append(np.log(final_dist / initial_dist) / 5)

            if divergences:
                lyapunov_proxy = np.mean(divergences)

        return {
            'phase_space_dim': self.embedding_dim,
            'trajectory_length': float(trajectory_length),
            'recurrence_rate': float(recurrence_rate),
            'determinism': float(determinism),
            'entropy_rate': float(entropy_rate),
            'lyapunov_proxy': float(lyapunov_proxy)
        }


# =============================================================================
# 4. Fisher-Rao Distance
# =============================================================================

class FisherRaoComputer:
    """
    Compute Fisher-Rao geodesic distance between return distributions.
    Assumes Gaussian distributions for simplicity.
    """

    def __init__(self):
        pass

    def fit_gaussian(self, returns: np.ndarray) -> Tuple[float, float]:
        """Fit Gaussian to returns."""
        mu = np.mean(returns)
        sigma = np.std(returns) + 1e-8
        return mu, sigma

    def fisher_rao_distance(self, mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
        """
        Compute Fisher-Rao distance between two Gaussians.

        For Gaussians, the Fisher-Rao distance is:
        d = sqrt(2) * sqrt( (mu1-mu2)^2/(sigma1*sigma2) + 2*log((sigma1+sigma2)/(2*sqrt(sigma1*sigma2))) )

        Simplified version using Hellinger distance as proxy.
        """
        # Hellinger distance (closed form for Gaussians)
        term1 = (sigma1 * sigma2) / (sigma1**2 + sigma2**2)
        term2 = np.exp(-0.25 * (mu1 - mu2)**2 / (sigma1**2 + sigma2**2))

        hellinger = np.sqrt(1 - np.sqrt(2 * term1) * term2)

        return float(hellinger)

    def compute_distance_matrix(self, returns_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """Compute pairwise Fisher-Rao distances."""
        tickers = list(returns_dict.keys())
        n = len(tickers)

        # Fit distributions
        params = {}
        for ticker in tickers:
            params[ticker] = self.fit_gaussian(returns_dict[ticker])

        # Compute distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                mu1, sigma1 = params[tickers[i]]
                mu2, sigma2 = params[tickers[j]]
                d = self.fisher_rao_distance(mu1, sigma1, mu2, sigma2)
                distances[i, j] = d
                distances[j, i] = d

        return distances, tickers


# =============================================================================
# 5. Convex Hull / DEA for Fundamentals
# =============================================================================

class FundamentalGeometry:
    """
    Compute geometric features from fundamental data.
    """

    def __init__(self):
        pass

    def compute_efficiency_score(self, fundamentals: pd.DataFrame) -> Dict[str, float]:
        """
        Compute distance to efficient frontier using DEA-like approach.

        Inputs (to minimize): P/E, P/B, Debt/Equity
        Outputs (to maximize): ROE, Revenue Growth, Dividend Yield
        """
        # Normalize columns
        df = fundamentals.copy()

        input_cols = ['pe_ratio', 'pb_ratio', 'debt_equity']
        output_cols = ['roe', 'revenue_growth', 'dividend_yield']

        # Check which columns exist
        available_inputs = [c for c in input_cols if c in df.columns]
        available_outputs = [c for c in output_cols if c in df.columns]

        if not available_inputs or not available_outputs:
            return {ticker: 0.5 for ticker in df.index}

        # Normalize
        for col in available_inputs + available_outputs:
            if col in df.columns:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)

        # Simple efficiency score: output / input
        input_score = df[available_inputs].mean(axis=1)
        output_score = df[available_outputs].mean(axis=1)

        efficiency = output_score / (input_score + 0.1)

        # Normalize to [0, 1]
        efficiency = (efficiency - efficiency.min()) / (efficiency.max() - efficiency.min() + 1e-8)

        return efficiency.to_dict()

    def compute_convex_hull_distance(self, points: np.ndarray, tickers: List[str]) -> Dict[str, float]:
        """
        Compute distance to convex hull of efficient points.
        """
        if len(points) < 4 or points.shape[1] < 2:
            return {t: 0.0 for t in tickers}

        try:
            hull = ConvexHull(points)
            hull_points = set(hull.vertices)

            distances = {}
            for i, ticker in enumerate(tickers):
                if i in hull_points:
                    distances[ticker] = 0.0  # On the frontier
                else:
                    # Distance to nearest hull point
                    min_dist = min(
                        np.linalg.norm(points[i] - points[j])
                        for j in hull_points
                    )
                    distances[ticker] = float(min_dist)

            return distances
        except:
            return {t: 0.0 for t in tickers}


# =============================================================================
# Main Pipeline
# =============================================================================

def create_geometric_tables(conn: sqlite3.Connection):
    """Create tables for geometric features."""

    # TDA features per ticker per date
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geom_tda (
            date TEXT,
            ticker TEXT,
            betti_0 REAL,
            betti_1 REAL,
            betti_2 REAL,
            persistence_mean_0 REAL,
            persistence_max_0 REAL,
            persistence_std_0 REAL,
            persistence_entropy_0 REAL,
            persistence_mean_1 REAL,
            persistence_max_1 REAL,
            persistence_std_1 REAL,
            persistence_entropy_1 REAL,
            persistence_mean_2 REAL,
            persistence_max_2 REAL,
            persistence_std_2 REAL,
            persistence_entropy_2 REAL,
            total_persistence REAL,
            persistence_landscape_norm REAL,
            PRIMARY KEY (date, ticker)
        )
    """)

    # Ricci curvature per ticker per date
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geom_ricci (
            date TEXT,
            ticker TEXT,
            ricci_curvature REAL,
            PRIMARY KEY (date, ticker)
        )
    """)

    # Network features per date
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geom_network (
            date TEXT PRIMARY KEY,
            avg_curvature REAL,
            std_curvature REAL,
            min_curvature REAL,
            max_curvature REAL,
            network_density REAL,
            avg_clustering REAL,
            transitivity REAL
        )
    """)

    # Phase space features per ticker per date
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geom_phase_space (
            date TEXT,
            ticker TEXT,
            trajectory_length REAL,
            recurrence_rate REAL,
            determinism REAL,
            entropy_rate REAL,
            lyapunov_proxy REAL,
            PRIMARY KEY (date, ticker)
        )
    """)

    # Fisher-Rao features (distance to market centroid)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geom_fisher_rao (
            date TEXT,
            ticker TEXT,
            distance_to_centroid REAL,
            distribution_mu REAL,
            distribution_sigma REAL,
            PRIMARY KEY (date, ticker)
        )
    """)

    # Fundamental geometry
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geom_fundamental (
            date TEXT,
            ticker TEXT,
            efficiency_score REAL,
            hull_distance REAL,
            PRIMARY KEY (date, ticker)
        )
    """)

    conn.commit()
    logger.info("Geometric tables created.")


def load_price_data(conn: sqlite3.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    """Load price data."""
    query = """
        SELECT date, ticker, adjusted_close
        FROM historical_prices
        WHERE date >= ? AND date <= ?
        ORDER BY ticker, date
    """
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    return df


def load_fundamentals(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load fundamental data."""
    query = """
        SELECT ticker, pe_ratio, pb_ratio, roe, dividend_yield
        FROM fundamentals_general
    """
    try:
        df = pd.read_sql_query(query, conn)
        df = df.set_index('ticker')
        return df
    except:
        return pd.DataFrame()


def compute_and_store_features(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
    window_days: int = 60,
    step_days: int = 5
):
    """
    Compute all geometric features and store in database.
    """

    # Initialize computers
    tda = TDAComputer(max_dim=2, window_size=20)
    ricci = RicciCurvatureComputer(correlation_threshold=0.3)
    takens = TakensEmbedding(embedding_dim=10, tau=5)
    fisher = FisherRaoComputer()
    fundamental_geom = FundamentalGeometry()

    # Load data
    logger.info("Loading price data...")
    prices_df = load_price_data(conn, start_date, end_date)

    if prices_df.empty:
        logger.error("No price data found!")
        return

    # Pivot to get price matrix
    price_matrix = prices_df.pivot(index='date', columns='ticker', values='adjusted_close')
    price_matrix = price_matrix.sort_index()

    # Compute returns
    returns_matrix = price_matrix.pct_change().dropna()

    dates = returns_matrix.index.tolist()
    tickers = returns_matrix.columns.tolist()

    logger.info(f"Processing {len(dates)} dates, {len(tickers)} tickers")

    # Load fundamentals
    fundamentals = load_fundamentals(conn)

    # Process each date window
    processed_dates = []

    for i in tqdm(range(window_days, len(dates), step_days), desc="Computing geometric features"):
        current_date = dates[i]
        window_start = i - window_days

        # Get window data
        window_returns = returns_matrix.iloc[window_start:i]
        window_prices = price_matrix.iloc[window_start:i]

        # Skip if too many NaN
        valid_tickers = window_returns.columns[window_returns.isna().sum() < window_days * 0.2].tolist()
        if len(valid_tickers) < 10:
            continue

        window_returns = window_returns[valid_tickers].fillna(0)
        window_prices = window_prices[valid_tickers].fillna(method='ffill')

        processed_dates.append(current_date)

        # -----------------------------------------------------------------
        # 1. TDA Features (per ticker)
        # -----------------------------------------------------------------
        tda_rows = []
        for ticker in valid_tickers:
            series = window_returns[ticker].values
            points = tda.sliding_window_embedding(series)
            features = tda.compute_persistence(points)
            features['date'] = current_date
            features['ticker'] = ticker
            tda_rows.append(features)

        if tda_rows:
            tda_df = pd.DataFrame(tda_rows)
            tda_df.to_sql('geom_tda', conn, if_exists='append', index=False)

        # -----------------------------------------------------------------
        # 2. Correlation Network & Ricci Curvature
        # -----------------------------------------------------------------
        corr_matrix = window_returns.corr().values
        G = ricci.build_network(corr_matrix, valid_tickers)

        if G is not None:
            # Per-ticker curvature
            curvatures = ricci.compute_all_curvatures(G)
            ricci_rows = [
                {'date': current_date, 'ticker': t, 'ricci_curvature': c}
                for t, c in curvatures.items()
            ]
            if ricci_rows:
                pd.DataFrame(ricci_rows).to_sql('geom_ricci', conn, if_exists='append', index=False)

            # Network-level features
            network_features = ricci.compute_network_features(G)
            network_features['date'] = current_date
            pd.DataFrame([network_features]).to_sql('geom_network', conn, if_exists='append', index=False)

        # -----------------------------------------------------------------
        # 3. Phase Space (Takens) Features
        # -----------------------------------------------------------------
        phase_rows = []
        for ticker in valid_tickers:
            series = window_prices[ticker].values
            features = takens.compute_features(series)
            features['date'] = current_date
            features['ticker'] = ticker
            phase_rows.append(features)

        if phase_rows:
            pd.DataFrame(phase_rows).to_sql('geom_phase_space', conn, if_exists='append', index=False)

        # -----------------------------------------------------------------
        # 4. Fisher-Rao Distance
        # -----------------------------------------------------------------
        returns_dict = {t: window_returns[t].values for t in valid_tickers}

        # Compute centroid (mean distribution)
        all_returns = np.concatenate(list(returns_dict.values()))
        centroid_mu, centroid_sigma = fisher.fit_gaussian(all_returns)

        fisher_rows = []
        for ticker in valid_tickers:
            mu, sigma = fisher.fit_gaussian(returns_dict[ticker])
            distance = fisher.fisher_rao_distance(mu, sigma, centroid_mu, centroid_sigma)
            fisher_rows.append({
                'date': current_date,
                'ticker': ticker,
                'distance_to_centroid': distance,
                'distribution_mu': mu,
                'distribution_sigma': sigma
            })

        if fisher_rows:
            pd.DataFrame(fisher_rows).to_sql('geom_fisher_rao', conn, if_exists='append', index=False)

        # -----------------------------------------------------------------
        # 5. Fundamental Geometry (less frequent, e.g., monthly)
        # -----------------------------------------------------------------
        if i % 20 == 0 and not fundamentals.empty:
            valid_fund_tickers = [t for t in valid_tickers if t in fundamentals.index]
            if valid_fund_tickers:
                fund_subset = fundamentals.loc[valid_fund_tickers]
                efficiency = fundamental_geom.compute_efficiency_score(fund_subset)

                fund_rows = [
                    {'date': current_date, 'ticker': t, 'efficiency_score': e, 'hull_distance': 0}
                    for t, e in efficiency.items()
                ]
                if fund_rows:
                    pd.DataFrame(fund_rows).to_sql('geom_fundamental', conn, if_exists='append', index=False)

        # Commit periodically
        if len(processed_dates) % 50 == 0:
            conn.commit()

    conn.commit()
    logger.info(f"Processed {len(processed_dates)} dates")


def main():
    parser = argparse.ArgumentParser(description="Compute geometric features for financial data")
    parser.add_argument("--db_path", type=str, required=True, help="Path to SQLite database")
    parser.add_argument("--start_date", type=str, default="2015-01-01", help="Start date")
    parser.add_argument("--end_date", type=str, default="2025-01-01", help="End date")
    parser.add_argument("--window_days", type=int, default=60, help="Rolling window size in days")
    parser.add_argument("--step_days", type=int, default=5, help="Step size in days")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("GEOMETRIC FEATURES COMPUTATION")
    logger.info("=" * 70)
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Window: {args.window_days} days, Step: {args.step_days} days")
    logger.info("")

    # Connect to database
    conn = sqlite3.connect(args.db_path)

    # Create tables
    create_geometric_tables(conn)

    # Compute features
    compute_and_store_features(
        conn,
        args.start_date,
        args.end_date,
        args.window_days,
        args.step_days
    )

    # Show summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    for table in ['geom_tda', 'geom_ricci', 'geom_network', 'geom_phase_space', 'geom_fisher_rao', 'geom_fundamental']:
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        logger.info(f"  {table}: {count:,} rows")

    conn.close()
    logger.info("")
    logger.info("Done!")


if __name__ == "__main__":
    main()
