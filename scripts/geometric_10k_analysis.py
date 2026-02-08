"""
Geometric Analysis of SEC 10-K Filings
======================================

Analyze geometric features from 10-K text embeddings:
1. Load 10-K embeddings from database
2. Compute TDA (persistent homology) on embedding space
3. Compute Fisher-Rao distances between company distributions
4. Build document manifold and analyze curvature
5. Test predictive power for stock returns

Usage:
    python scripts/geometric_10k_analysis.py --db_path data/eodhd_sp500.db
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
import json
warnings.filterwarnings('ignore')

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("Warning: ripser not installed. Using simplified TDA.")

from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import spearmanr, pearsonr, entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# 10-K Embedding Loaders
# =============================================================================

def load_10k_embeddings(db_path: str) -> pd.DataFrame:
    """Load 10-K embeddings from database."""
    conn = sqlite3.connect(db_path)

    # Check what tables exist
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"Available tables: {tables}")

    # Try different possible table names
    embedding_tables = ['sec_10k_embeddings', 'mamba_features_daily', 'news_embeddings_daily']

    df = None
    for table in embedding_tables:
        if table in tables:
            try:
                # Get table structure
                cursor = conn.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                print(f"\n{table} columns: {columns}")

                # Load data
                if 'embedding' in columns:
                    query = f"SELECT ticker, date, embedding FROM {table} LIMIT 10"
                    sample = pd.read_sql_query(query, conn)
                    print(f"Sample from {table}:")
                    print(sample.head(2))

                    # Load full data
                    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                    print(f"\nLoaded {len(df)} rows from {table}")
                    break
            except Exception as e:
                print(f"Error loading {table}: {e}")
                continue

    conn.close()
    return df


def load_10k_content(db_path: str, limit: int = 500) -> pd.DataFrame:
    """Load raw 10-K content for text analysis."""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT ticker, filing_date, content
        FROM sec_filings
        WHERE filing_type = '10-K' AND content IS NOT NULL
        LIMIT ?
    """

    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()

    print(f"Loaded {len(df)} 10-K filings")
    return df


def load_price_data(db_path: str) -> pd.DataFrame:
    """Load price data for return calculation."""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT DATE(date) as date, ticker, adjusted_close
        FROM historical_prices
        WHERE adjusted_close > 0
        ORDER BY ticker, date
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


# =============================================================================
# Text-based Geometric Features
# =============================================================================

def extract_text_features(content: str) -> Dict[str, float]:
    """Extract simple text features for geometric analysis."""
    if not content or len(content) < 100:
        return None

    # Word count
    words = content.lower().split()
    n_words = len(words)

    # Risk-related keywords
    risk_words = ['risk', 'uncertain', 'volatile', 'decline', 'adverse', 'challenging',
                  'litigation', 'regulatory', 'competition', 'threat', 'loss']
    risk_count = sum(1 for w in words if any(r in w for r in risk_words))
    risk_density = risk_count / n_words if n_words > 0 else 0

    # Growth-related keywords
    growth_words = ['growth', 'increase', 'expand', 'opportunity', 'improve', 'strong',
                    'innovation', 'invest', 'develop', 'strategic']
    growth_count = sum(1 for w in words if any(g in w for g in growth_words))
    growth_density = growth_count / n_words if n_words > 0 else 0

    # Financial keywords
    financial_words = ['revenue', 'profit', 'margin', 'earnings', 'cash', 'debt',
                       'capital', 'asset', 'liability', 'equity']
    financial_count = sum(1 for w in words if any(f in w for f in financial_words))
    financial_density = financial_count / n_words if n_words > 0 else 0

    # Sentence statistics (proxy for complexity)
    sentences = content.split('.')
    n_sentences = len([s for s in sentences if len(s.strip()) > 0])
    avg_sentence_length = n_words / n_sentences if n_sentences > 0 else 0

    # Unique word ratio (vocabulary diversity)
    unique_words = len(set(words))
    vocab_diversity = unique_words / n_words if n_words > 0 else 0

    # Number statistics (more numbers = more quantitative)
    import re
    numbers = re.findall(r'\d+\.?\d*', content)
    number_density = len(numbers) / n_words if n_words > 0 else 0

    return {
        'n_words': n_words,
        'risk_density': risk_density,
        'growth_density': growth_density,
        'financial_density': financial_density,
        'avg_sentence_length': avg_sentence_length,
        'vocab_diversity': vocab_diversity,
        'number_density': number_density,
        'sentiment_proxy': growth_density - risk_density  # Simple sentiment
    }


def create_text_embedding(features: Dict[str, float]) -> np.ndarray:
    """Create embedding vector from text features."""
    return np.array([
        features['risk_density'],
        features['growth_density'],
        features['financial_density'],
        features['avg_sentence_length'] / 100,  # Normalize
        features['vocab_diversity'],
        features['number_density'],
        features['sentiment_proxy'],
        np.log1p(features['n_words']) / 10  # Log-normalized word count
    ])


# =============================================================================
# Geometric Analysis Functions
# =============================================================================

def compute_embedding_tda(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute TDA features on embedding space."""
    if len(embeddings) < 10:
        return {'betti_0': 0, 'betti_1': 0, 'persistence_entropy': 0}

    # Normalize embeddings
    embeddings = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-8)

    if HAS_RIPSER:
        try:
            result = ripser(embeddings, maxdim=1, thresh=3.0)
            diagrams = result['dgms']

            betti_0 = len(diagrams[0])
            betti_1 = len([p for p in diagrams[1] if np.isfinite(p[1])])

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
            else:
                pers_entropy = 0

            return {
                'betti_0': betti_0,
                'betti_1': betti_1,
                'persistence_entropy': pers_entropy
            }
        except:
            pass

    # Fallback
    dists = pdist(embeddings)
    return {
        'betti_0': len(embeddings),
        'betti_1': 0,
        'persistence_entropy': entropy(np.histogram(dists, bins=20)[0] + 1)
    }


def compute_sector_manifold(embeddings: np.ndarray, sectors: List[str]) -> Dict[str, float]:
    """Compute manifold features by sector."""
    unique_sectors = list(set(sectors))

    sector_centroids = {}
    sector_spreads = {}

    for sector in unique_sectors:
        mask = np.array([s == sector for s in sectors])
        if mask.sum() < 3:
            continue

        sector_emb = embeddings[mask]
        centroid = sector_emb.mean(axis=0)
        spread = np.mean([np.linalg.norm(e - centroid) for e in sector_emb])

        sector_centroids[sector] = centroid
        sector_spreads[sector] = spread

    # Inter-sector distances
    if len(sector_centroids) >= 2:
        centroids_array = np.array(list(sector_centroids.values()))
        inter_distances = pdist(centroids_array)
        avg_inter_distance = np.mean(inter_distances)
    else:
        avg_inter_distance = 0

    # Average intra-sector spread
    avg_spread = np.mean(list(sector_spreads.values())) if sector_spreads else 0

    return {
        'n_sectors': len(sector_centroids),
        'avg_inter_sector_distance': avg_inter_distance,
        'avg_intra_sector_spread': avg_spread,
        'separation_ratio': avg_inter_distance / (avg_spread + 1e-8)
    }


def compute_similarity_network(embeddings: np.ndarray, threshold: float = 0.7) -> Dict[str, float]:
    """Compute network features from similarity matrix."""
    # Cosine similarity
    sim_matrix = cosine_similarity(embeddings)

    # Create adjacency matrix
    adj = (sim_matrix > threshold).astype(float)
    np.fill_diagonal(adj, 0)

    # Network statistics
    n_nodes = len(embeddings)
    n_edges = adj.sum() / 2
    density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0

    # Degree distribution
    degrees = adj.sum(axis=1)
    avg_degree = degrees.mean()
    degree_std = degrees.std()

    # Clustering coefficient (simplified)
    clustering = 0
    for i in range(n_nodes):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) >= 2:
            submatrix = adj[np.ix_(neighbors, neighbors)]
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            actual_edges = submatrix.sum() / 2
            clustering += actual_edges / possible_edges if possible_edges > 0 else 0
    clustering /= n_nodes

    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': density,
        'avg_degree': avg_degree,
        'degree_std': degree_std,
        'clustering': clustering
    }


def compute_filing_distance_to_peers(
    embedding: np.ndarray,
    sector_embeddings: np.ndarray
) -> float:
    """Compute distance from filing to sector peers."""
    if len(sector_embeddings) < 2:
        return 0

    # Centroid of peers
    centroid = sector_embeddings.mean(axis=0)

    # Distance to centroid
    distance = np.linalg.norm(embedding - centroid)

    # Normalize by average peer distance
    avg_peer_dist = np.mean([np.linalg.norm(e - centroid) for e in sector_embeddings])

    return distance / (avg_peer_dist + 1e-8)


# =============================================================================
# Alpha Analysis
# =============================================================================

def compute_future_returns(
    prices_df: pd.DataFrame,
    ticker: str,
    date: str,
    horizons: List[int] = [5, 20, 60]
) -> Dict[str, float]:
    """Compute future returns for various horizons."""
    ticker_prices = prices_df[prices_df['ticker'] == ticker].copy()
    if len(ticker_prices) < 100:
        return {f'return_{h}d': None for h in horizons}

    ticker_prices = ticker_prices.sort_values('date')
    dates = ticker_prices['date'].tolist()

    # Find closest date
    try:
        idx = min(range(len(dates)), key=lambda i: abs(
            (pd.to_datetime(dates[i]) - pd.to_datetime(date)).days
        ))
    except:
        return {f'return_{h}d': None for h in horizons}

    current_price = ticker_prices.iloc[idx]['adjusted_close']

    returns = {}
    for h in horizons:
        if idx + h < len(ticker_prices):
            future_price = ticker_prices.iloc[idx + h]['adjusted_close']
            returns[f'return_{h}d'] = (future_price / current_price - 1) * 100
        else:
            returns[f'return_{h}d'] = None

    return returns


def analyze_10k_alpha(results_df: pd.DataFrame):
    """Analyze predictive power of 10-K geometric features."""
    print("\n" + "="*70)
    print("10-K GEOMETRIC FEATURES vs FUTURE RETURNS")
    print("="*70)

    if len(results_df) == 0:
        print("No data to analyze!")
        return pd.DataFrame()

    features = [col for col in results_df.columns if col not in
                ['ticker', 'date', 'sector', 'return_5d', 'return_20d', 'return_60d']]

    horizons = ['return_5d', 'return_20d', 'return_60d']

    correlations = []

    for feature in features:
        for horizon in horizons:
            if horizon not in results_df.columns:
                continue

            mask = results_df[feature].notna() & results_df[horizon].notna()
            x = results_df.loc[mask, feature].values
            y = results_df.loc[mask, horizon].values

            if len(x) < 20:
                continue

            r, p = spearmanr(x, y)

            correlations.append({
                'feature': feature,
                'horizon': horizon,
                'correlation': r,
                'p_value': p,
                'significant': p < 0.05,
                'n_samples': len(x)
            })

    corr_df = pd.DataFrame(correlations)

    if len(corr_df) > 0:
        print("\nAll correlations:")
        print("-" * 70)
        print(f"{'Feature':<30} {'Horizon':<12} {'Corr':>8} {'p-val':>10} {'Sig':>5}")
        print("-" * 70)

        for _, row in corr_df.sort_values('p_value').iterrows():
            sig = "*" if row['significant'] else ""
            print(f"{row['feature']:<30} {row['horizon']:<12} "
                  f"{row['correlation']:>+8.3f} {row['p_value']:>10.4f} {sig:>5}")

        # Significant findings
        sig_df = corr_df[corr_df['significant']]
        if len(sig_df) > 0:
            print(f"\n Found {len(sig_df)} significant correlations (p < 0.05)")
        else:
            print("\n No significant correlations at p < 0.05")

    return corr_df


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Geometric Analysis of 10-K Filings")
    parser.add_argument("--db_path", type=str, required=True, help="Path to database")
    parser.add_argument("--limit", type=int, default=500, help="Max filings to process")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    print("="*70)
    print("GEOMETRIC ANALYSIS OF SEC 10-K FILINGS")
    print("="*70)
    print(f"Database: {args.db_path}")
    print()

    # 1. Load 10-K content
    print("Loading 10-K filings...")
    filings_df = load_10k_content(args.db_path, limit=args.limit)

    if len(filings_df) == 0:
        print("No 10-K filings found!")
        return

    # 2. Load price data
    print("\nLoading price data...")
    prices_df = load_price_data(args.db_path)
    print(f"Loaded {len(prices_df)} price records")

    # 3. Load sector information
    conn = sqlite3.connect(args.db_path)
    sectors_df = pd.read_sql_query(
        "SELECT ticker, sector FROM fundamentals_general", conn
    )
    conn.close()
    ticker_to_sector = dict(zip(sectors_df['ticker'], sectors_df['sector']))

    # 4. Extract text features and compute embeddings
    print("\nExtracting text features...")
    results = []
    embeddings = []
    tickers = []
    sectors = []

    for _, row in tqdm(filings_df.iterrows(), total=len(filings_df)):
        ticker = row['ticker']
        date = row['filing_date']
        content = row['content']

        # Extract text features
        features = extract_text_features(content)
        if features is None:
            continue

        # Create embedding
        embedding = create_text_embedding(features)

        # Get sector
        sector = ticker_to_sector.get(ticker, 'Unknown')

        # Get future returns
        returns = compute_future_returns(prices_df, ticker, date)

        result = {
            'ticker': ticker,
            'date': date,
            'sector': sector,
            **features,
            **returns
        }
        results.append(result)
        embeddings.append(embedding)
        tickers.append(ticker)
        sectors.append(sector)

    results_df = pd.DataFrame(results)
    embeddings = np.array(embeddings)

    print(f"\nProcessed {len(results_df)} filings with features")

    # 5. Compute global geometric features
    print("\nComputing geometric features on embedding space...")

    # TDA on full embedding space
    tda_features = compute_embedding_tda(embeddings)
    print(f"TDA: Betti-0={tda_features['betti_0']}, Betti-1={tda_features['betti_1']}, "
          f"Persistence Entropy={tda_features['persistence_entropy']:.3f}")

    # Sector manifold analysis
    manifold_features = compute_sector_manifold(embeddings, sectors)
    print(f"Sector Manifold: {manifold_features['n_sectors']} sectors, "
          f"Separation ratio={manifold_features['separation_ratio']:.3f}")

    # Similarity network
    network_features = compute_similarity_network(embeddings, threshold=0.7)
    print(f"Similarity Network: {int(network_features['n_edges'])} edges, "
          f"Clustering={network_features['clustering']:.3f}")

    # 6. Compute per-filing geometric features
    print("\nComputing per-filing geometric features...")

    peer_distances = []
    for i, (ticker, sector) in enumerate(zip(tickers, sectors)):
        # Get peer embeddings
        peer_mask = np.array([s == sector and t != ticker for t, s in zip(tickers, sectors)])
        if peer_mask.sum() > 0:
            peer_emb = embeddings[peer_mask]
            dist = compute_filing_distance_to_peers(embeddings[i], peer_emb)
        else:
            dist = 0
        peer_distances.append(dist)

    results_df['distance_to_peers'] = peer_distances

    # 7. Analyze alpha
    corr_df = analyze_10k_alpha(results_df)

    # 8. Visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. PCA of embeddings colored by sector
    ax = axes[0, 0]
    if len(embeddings) > 10:
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(embeddings)

        unique_sectors = list(set(sectors))[:10]  # Top 10 sectors
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sectors)))

        for i, sector in enumerate(unique_sectors):
            mask = np.array([s == sector for s in sectors])
            if mask.sum() > 0:
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                          c=[colors[i]], label=sector[:15], alpha=0.6, s=30)

        ax.set_title('10-K Embedding Space (PCA)')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.legend(loc='best', fontsize=6)

    # 2. Risk vs Growth density
    ax = axes[0, 1]
    if 'risk_density' in results_df.columns and 'growth_density' in results_df.columns:
        ax.scatter(results_df['risk_density'], results_df['growth_density'],
                  alpha=0.5, s=20)
        ax.set_xlabel('Risk Keyword Density')
        ax.set_ylabel('Growth Keyword Density')
        ax.set_title('Risk vs Growth in 10-K Language')

        # Add quadrant labels
        ax.axhline(y=results_df['growth_density'].median(), color='r', linestyle='--', alpha=0.3)
        ax.axvline(x=results_df['risk_density'].median(), color='r', linestyle='--', alpha=0.3)

    # 3. Distance to peers vs returns
    ax = axes[0, 2]
    if 'return_20d' in results_df.columns:
        mask = results_df['return_20d'].notna()
        ax.scatter(results_df.loc[mask, 'distance_to_peers'],
                  results_df.loc[mask, 'return_20d'], alpha=0.5, s=20)
        ax.set_xlabel('Distance to Sector Peers')
        ax.set_ylabel('20-day Return (%)')
        ax.set_title('Peer Distance vs Returns')

    # 4. Sentiment proxy vs returns
    ax = axes[1, 0]
    if 'sentiment_proxy' in results_df.columns and 'return_20d' in results_df.columns:
        mask = results_df['return_20d'].notna()
        ax.scatter(results_df.loc[mask, 'sentiment_proxy'],
                  results_df.loc[mask, 'return_20d'], alpha=0.5, s=20, c='green')
        ax.set_xlabel('Sentiment Proxy (Growth - Risk)')
        ax.set_ylabel('20-day Return (%)')
        ax.set_title('10-K Sentiment vs Returns')

        # Trend line
        x = results_df.loc[mask, 'sentiment_proxy'].values
        y = results_df.loc[mask, 'return_20d'].values
        if len(x) > 5:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=2,
                   label=f'Trend: {z[0]:.1f}x + {z[1]:.1f}')
            ax.legend()

    # 5. Sector average sentiment
    ax = axes[1, 1]
    if 'sentiment_proxy' in results_df.columns:
        sector_sentiment = results_df.groupby('sector')['sentiment_proxy'].mean().sort_values()
        sector_sentiment = sector_sentiment[sector_sentiment.index != 'Unknown'][-15:]  # Top 15

        colors = ['green' if v > 0 else 'red' for v in sector_sentiment.values]
        ax.barh(range(len(sector_sentiment)), sector_sentiment.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(sector_sentiment)))
        ax.set_yticklabels([s[:20] for s in sector_sentiment.index], fontsize=8)
        ax.set_xlabel('Average Sentiment Proxy')
        ax.set_title('Sector Sentiment from 10-K')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

    # 6. Correlation heatmap
    ax = axes[1, 2]
    if len(corr_df) > 0:
        pivot = corr_df.pivot_table(
            index='feature',
            columns='horizon',
            values='correlation'
        )

        # Simple heatmap
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title('Feature-Return Correlations')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    output_path = f"{args.output_dir}/geometric_10k_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")

    # Save results
    output_csv = f"{args.output_dir}/geometric_10k_features.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"Saved features to: {output_csv}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Processed {len(results_df)} 10-K filings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"\nGlobal Geometric Features:")
    print(f"  - TDA Betti-1 (cycles in manifold): {tda_features['betti_1']}")
    print(f"  - Persistence Entropy: {tda_features['persistence_entropy']:.3f}")
    print(f"  - Sector Separation Ratio: {manifold_features['separation_ratio']:.3f}")
    print(f"  - Similarity Network Clustering: {network_features['clustering']:.3f}")

    # Key findings
    if len(corr_df) > 0:
        sig = corr_df[corr_df['significant']]
        if len(sig) > 0:
            print(f"\n SIGNIFICANT PREDICTORS FOUND:")
            for _, row in sig.sort_values('p_value').head(5).iterrows():
                print(f"  - {row['feature']} -> {row['horizon']}: r={row['correlation']:.3f}")

    plt.show()


if __name__ == "__main__":
    main()
