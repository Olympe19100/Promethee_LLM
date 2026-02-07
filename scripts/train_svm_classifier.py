"""
Train SVM Classifier on Promethee Embeddings

After training Promethee, this script:
1. Extracts embeddings for all training samples
2. Trains an SVM to classify winners vs losers
3. Saves the SVM for inference

Usage:
    python scripts/train_svm_classifier.py \
        --model_path models/promethee_semantic \
        --data_path data/promethee_semantic.jsonl \
        --output_path models/promethee_svm.joblib \
        --max_samples 50000
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tqdm import tqdm
from pathlib import Path
from loguru import logger

from transformers import AutoTokenizer, MambaForCausalLM


# =============================================================================
# Promethee Model (same as training)
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

        result = {'loss': outputs.loss, 'logits': outputs.logits}

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
            result['direction_logits'] = self.direction_head(embeddings)
            result['confidence_logits'] = self.confidence_head(embeddings)
            result['risk_logits'] = self.risk_head(embeddings)

        return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SVM on Promethee embeddings")
    parser.add_argument("--model_path", type=str, default="models/promethee_semantic",
                        help="Path to trained Promethee model")
    parser.add_argument("--data_path", type=str, default="data/promethee_semantic.jsonl",
                        help="Path to training data")
    parser.add_argument("--output_path", type=str, default="models/promethee_svm.joblib",
                        help="Output path for SVM")
    parser.add_argument("--max_samples", type=int, default=50000,
                        help="Max samples to use")
    parser.add_argument("--return_threshold", type=float, default=0.01,
                        help="Return threshold for winner/loser (default 1%)")
    parser.add_argument("--kernel", type=str, default="rbf",
                        choices=["linear", "rbf", "poly"],
                        help="SVM kernel")
    parser.add_argument("--C", type=float, default=1.0,
                        help="SVM regularization")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for embedding extraction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # =========================================================================
    # Load Promethee Model
    # =========================================================================
    logger.info(f"Loading Promethee from {args.model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = MambaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16
    ).to(device)

    # Load checkpoint with embedding head
    checkpoint_path = Path(args.model_path) / "promethee_final.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        embed_dim = checkpoint.get('embed_dim', 256)
        model = PrometheeModel(base_model, embed_dim=embed_dim).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint with embed_dim={embed_dim}")
    else:
        # Fallback: just use base model with new embedding head
        logger.warning("No checkpoint found, using base model with new embedding head")
        model = PrometheeModel(base_model, embed_dim=256).to(device)

    model.eval()

    # =========================================================================
    # Load Data
    # =========================================================================
    logger.info(f"Loading data from {args.data_path}...")

    samples = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(samples)} samples")

    # =========================================================================
    # Extract Embeddings
    # =========================================================================
    logger.info("Extracting embeddings...")

    X = []
    y = []
    metadata = []

    for sample in tqdm(samples, desc="Extracting embeddings"):
        # Get return
        ret_1d = sample['ground_truth']['return_1d'] / 100

        # Label
        if ret_1d > args.return_threshold:
            label = 1  # Winner
        elif ret_1d < -args.return_threshold:
            label = -1  # Loser
        else:
            continue  # Skip neutral

        # Tokenize input
        inputs = tokenizer(
            sample['input'][:2048],
            return_tensors='pt',
            truncation=True,
            max_length=2048,
            padding='max_length'
        )

        # Extract embedding
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                return_embeddings=True
            )
            embedding = outputs['embeddings'].cpu().numpy().flatten()

        X.append(embedding)
        y.append(label)
        metadata.append({
            'ticker': sample.get('ticker'),
            'date': sample.get('date'),
            'return_1d': ret_1d
        })

    X = np.array(X)
    y = np.array(y)

    logger.info(f"Dataset: {len(X)} samples")
    logger.info(f"  Winners (+1): {sum(y == 1)}")
    logger.info(f"  Losers (-1): {sum(y == -1)}")
    logger.info(f"  Embedding dim: {X.shape[1]}")

    # =========================================================================
    # Train/Test Split
    # =========================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # =========================================================================
    # Scale Features
    # =========================================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================================================================
    # Train SVM
    # =========================================================================
    logger.info(f"Training SVM (kernel={args.kernel}, C={args.C})...")

    base_svm = SVC(
        kernel=args.kernel,
        C=args.C,
        gamma='scale',
        class_weight='balanced',
        random_state=42,
        probability=False  # Will use CalibratedClassifierCV
    )

    # Cross-validation score
    cv_scores = cross_val_score(base_svm, X_train_scaled, y_train, cv=5)
    logger.info(f"Cross-val accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

    # Train with calibration for probabilities
    svm = CalibratedClassifierCV(base_svm, cv=5)
    svm.fit(X_train_scaled, y_train)

    # =========================================================================
    # Evaluate
    # =========================================================================
    logger.info("Evaluating on test set...")

    y_pred = svm.predict(X_test_scaled)
    y_proba = svm.predict_proba(X_test_scaled)

    # Classification report
    logger.info("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Loser', 'Winner']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # Precision for trading
    if cm[1,1] + cm[0,1] > 0:
        precision = cm[1,1] / (cm[1,1] + cm[0,1])
        logger.info(f"\nPrecision (when predicting winner): {precision:.2%}")

    # =========================================================================
    # Analyze Score Distribution
    # =========================================================================
    scores = y_proba[:, 1] - y_proba[:, 0]  # Score in [-1, +1]

    logger.info("\nScore distribution:")
    logger.info(f"  Min: {scores.min():.3f}")
    logger.info(f"  Max: {scores.max():.3f}")
    logger.info(f"  Mean: {scores.mean():.3f}")
    logger.info(f"  Std: {scores.std():.3f}")

    # Top predictions accuracy
    top_k = 100
    top_indices = np.argsort(scores)[-top_k:]
    top_accuracy = (y_test[top_indices] == 1).mean()
    logger.info(f"\nTop-{top_k} predictions accuracy: {top_accuracy:.2%}")

    bottom_k = 100
    bottom_indices = np.argsort(scores)[:bottom_k]
    bottom_accuracy = (y_test[bottom_indices] == -1).mean()
    logger.info(f"Bottom-{bottom_k} predictions accuracy: {bottom_accuracy:.2%}")

    # =========================================================================
    # Save
    # =========================================================================
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        'svm': svm,
        'scaler': scaler,
        'embed_dim': X.shape[1],
        'return_threshold': args.return_threshold,
        'kernel': args.kernel,
        'C': args.C,
        'cv_accuracy': cv_scores.mean(),
        'test_accuracy': (y_pred == y_test).mean(),
        'top_100_accuracy': top_accuracy
    }, output_path)

    logger.info(f"\nSVM saved to: {output_path}")

    # =========================================================================
    # Save thresholds for inference
    # =========================================================================
    config = {
        'model_path': args.model_path,
        'svm_path': str(output_path),
        'embed_dim': int(X.shape[1]),
        'return_threshold': args.return_threshold,
        'score_thresholds': {
            'strong_long': float(np.percentile(scores[y_test == 1], 75)),
            'long': float(np.percentile(scores[y_test == 1], 50)),
            'neutral_high': float(np.percentile(scores, 60)),
            'neutral_low': float(np.percentile(scores, 40)),
            'short': float(np.percentile(scores[y_test == -1], 50)),
            'strong_short': float(np.percentile(scores[y_test == -1], 25))
        },
        'performance': {
            'cv_accuracy': float(cv_scores.mean()),
            'test_accuracy': float((y_pred == y_test).mean()),
            'top_100_accuracy': float(top_accuracy),
            'bottom_100_accuracy': float(bottom_accuracy)
        }
    }

    config_path = output_path.with_suffix('.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Config saved to: {config_path}")


if __name__ == "__main__":
    main()
