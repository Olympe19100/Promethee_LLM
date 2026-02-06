"""
Generate SOTA News Embeddings using Qwen3-Embedding-8B

This script processes all news articles from the database and generates
high-quality embeddings using the SOTA Qwen3-Embedding model.

Usage on RunPod (A100 40GB recommended):
    python scripts/generate_news_embeddings.py \
        --db_path /workspace/data/eodhd_sp500.db \
        --output_dir /workspace/embeddings \
        --model Alibaba-NLP/gte-Qwen2-7B-instruct \
        --batch_size 32

For even better quality (requires more VRAM):
    python scripts/generate_news_embeddings.py \
        --model intfloat/e5-mistral-7b-instruct \
        --batch_size 16
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger


# SOTA Embedding Models (ranked by MTEB score)
EMBEDDING_MODELS = {
    "qwen3-8b": "Alibaba-NLP/gte-Qwen2-7B-instruct",  # 4096-dim, SOTA
    "e5-mistral": "intfloat/e5-mistral-7b-instruct",  # 4096-dim, excellent
    "bge-m3": "BAAI/bge-m3",  # 1024-dim, fast & good
    "e5-large": "intfloat/e5-large-v2",  # 1024-dim, good balance
    "gte-large": "Alibaba-NLP/gte-large-en-v1.5",  # 1024-dim
}

DEFAULT_MODEL = "qwen3-8b"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512


class NewsEmbeddingGenerator:
    """Generate embeddings for news articles using SOTA models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        max_length: int = DEFAULT_MAX_LENGTH,
        use_fp16: bool = True,
    ):
        self.device = device
        self.max_length = max_length
        self.use_fp16 = use_fp16

        # Resolve model name
        if model_name in EMBEDDING_MODELS:
            self.model_id = EMBEDDING_MODELS[model_name]
        else:
            self.model_id = model_name

        logger.info(f"Loading embedding model: {self.model_id}")
        self._load_model()

    def _load_model(self):
        """Load the embedding model and tokenizer."""
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        # Load model
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device == "cuda" and not hasattr(self.model, 'hf_device_map'):
            self.model = self.model.to(self.device)

        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Model loaded: {self.embedding_dim}-dim embeddings")

    def _format_text(self, title: str, content: str = None) -> str:
        """Format news text for embedding."""
        if content:
            text = f"{title}. {content}"
        else:
            text = title

        # Truncate if needed (will be done by tokenizer anyway)
        return text[:10000]

    @torch.no_grad()
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        # For instruction-tuned models, add task prefix
        if "instruct" in self.model_id.lower() or "e5" in self.model_id.lower():
            # E5/GTE instruction format
            texts = [f"Instruct: Analyze this financial news for sentiment and relevance.\nQuery: {t}" for t in texts]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)

        # Pool embeddings (use CLS token or mean pooling)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            # Mean pooling over sequence
            attention_mask = inputs['attention_mask']
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            embeddings = sum_embeddings / sum_mask

        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)


def load_news_from_db(
    db_path: str,
    start_date: str = "2000-01-01",
    end_date: str = "2030-12-31",
    limit: Optional[int] = None
) -> List[Dict]:
    """Load news articles from database."""
    conn = sqlite3.connect(db_path)

    query = """
        SELECT id, date, ticker, title, content
        FROM news
        WHERE date >= ? AND date <= ?
        ORDER BY date, ticker
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query, (start_date, end_date))

    articles = []
    for row in cursor:
        articles.append({
            "id": row[0],
            "date": row[1][:10] if row[1] else None,
            "ticker": row[2],
            "title": row[3] or "",
            "content": row[4] or ""
        })

    conn.close()
    return articles


def save_embeddings_batch(
    embeddings: np.ndarray,
    metadata: List[Dict],
    output_dir: Path,
    batch_idx: int
):
    """Save a batch of embeddings to disk."""
    # Save embeddings as numpy
    np.save(output_dir / f"embeddings_{batch_idx:06d}.npy", embeddings)

    # Save metadata as JSONL
    with open(output_dir / f"metadata_{batch_idx:06d}.jsonl", 'w') as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")


def create_aggregated_output(output_dir: Path, output_path: Path):
    """Aggregate all batches into a single file."""
    logger.info("Aggregating embeddings...")

    # Find all batch files
    embedding_files = sorted(output_dir.glob("embeddings_*.npy"))
    metadata_files = sorted(output_dir.glob("metadata_*.jsonl"))

    all_embeddings = []
    all_metadata = []

    for emb_file, meta_file in tqdm(zip(embedding_files, metadata_files), desc="Aggregating"):
        all_embeddings.append(np.load(emb_file))
        with open(meta_file, 'r') as f:
            for line in f:
                all_metadata.append(json.loads(line))

    # Stack embeddings
    embeddings = np.vstack(all_embeddings)

    # Save final output
    logger.info(f"Saving {len(embeddings)} embeddings to {output_path}")
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        metadata=np.array([json.dumps(m) for m in all_metadata], dtype=object)
    )

    # Also save index for fast lookup
    index = {m["id"]: i for i, m in enumerate(all_metadata)}
    with open(output_path.with_suffix('.index.json'), 'w') as f:
        json.dump(index, f)

    logger.info(f"Final shape: {embeddings.shape}")


def main():
    parser = argparse.ArgumentParser(description="Generate news embeddings with SOTA models")
    parser.add_argument("--db_path", type=str, required=True, help="Path to SQLite database")
    parser.add_argument("--output_dir", type=str, default="./embeddings", help="Output directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=list(EMBEDDING_MODELS.keys()) + ["custom"],
                        help="Embedding model to use")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Custom HuggingFace model ID (if --model=custom)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--start_date", type=str, default="2000-01-01")
    parser.add_argument("--end_date", type=str, default="2030-12-31")
    parser.add_argument("--limit", type=int, default=None, help="Max articles to process")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--save_every", type=int, default=10000, help="Save checkpoint every N articles")
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model
    model_name = args.model_id if args.model == "custom" else args.model

    # Log config
    logger.info("=" * 60)
    logger.info("News Embedding Generation")
    logger.info("=" * 60)
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info("")

    # Load articles
    logger.info("Loading news articles from database...")
    articles = load_news_from_db(
        args.db_path,
        args.start_date,
        args.end_date,
        args.limit
    )
    logger.info(f"Loaded {len(articles):,} articles")

    if not articles:
        logger.error("No articles found!")
        return

    # Resume handling
    start_idx = 0
    if args.resume:
        existing_batches = list(output_dir.glob("embeddings_*.npy"))
        if existing_batches:
            last_batch = max(int(f.stem.split('_')[1]) for f in existing_batches)
            start_idx = (last_batch + 1) * args.save_every
            logger.info(f"Resuming from index {start_idx}")

    # Initialize model
    generator = NewsEmbeddingGenerator(
        model_name=model_name,
        device=args.device,
        max_length=args.max_length
    )

    # Process articles in batches
    batch_embeddings = []
    batch_metadata = []
    batch_idx = start_idx // args.save_every

    for i in tqdm(range(start_idx, len(articles)), desc="Generating embeddings"):
        article = articles[i]

        # Format text
        text = generator._format_text(article["title"], article["content"])
        batch_embeddings.append(text)
        batch_metadata.append({
            "id": article["id"],
            "date": article["date"],
            "ticker": article["ticker"]
        })

        # Process batch
        if len(batch_embeddings) >= args.batch_size:
            embeddings = generator.embed_batch(batch_embeddings)

            # Accumulate for checkpoint
            if 'checkpoint_embeddings' not in locals():
                checkpoint_embeddings = []
                checkpoint_metadata = []

            checkpoint_embeddings.append(embeddings)
            checkpoint_metadata.extend(batch_metadata)

            batch_embeddings = []
            batch_metadata = []

            # Save checkpoint
            if len(checkpoint_metadata) >= args.save_every:
                all_emb = np.vstack(checkpoint_embeddings)
                save_embeddings_batch(all_emb, checkpoint_metadata, output_dir, batch_idx)
                logger.info(f"Saved checkpoint {batch_idx}: {len(checkpoint_metadata)} embeddings")

                checkpoint_embeddings = []
                checkpoint_metadata = []
                batch_idx += 1

    # Process remaining
    if batch_embeddings:
        embeddings = generator.embed_batch(batch_embeddings)
        if 'checkpoint_embeddings' not in locals():
            checkpoint_embeddings = []
            checkpoint_metadata = []
        checkpoint_embeddings.append(embeddings)
        checkpoint_metadata.extend(batch_metadata)

    # Save final checkpoint
    if 'checkpoint_embeddings' in locals() and checkpoint_embeddings:
        all_emb = np.vstack(checkpoint_embeddings)
        save_embeddings_batch(all_emb, checkpoint_metadata, output_dir, batch_idx)
        logger.info(f"Saved final checkpoint {batch_idx}: {len(checkpoint_metadata)} embeddings")

    # Aggregate all batches
    final_output = output_dir / "news_embeddings.npz"
    create_aggregated_output(output_dir, final_output)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info(f"Output: {final_output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
