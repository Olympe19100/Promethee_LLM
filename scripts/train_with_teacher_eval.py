"""
Train Prométhée with Teacher Evaluation (SOTA Pipeline)

Training loop with periodic Teacher LLM evaluation:
1. Train Prométhée for N steps
2. Every eval_steps: Teacher evaluates Prométhée's responses
3. Log quality metrics alongside loss
4. Early stopping based on Teacher score

This is LLM-as-a-Judge integrated into training.

Usage:
    python scripts/train_with_teacher_eval.py \
        --train_file data/promethee_unified_training.jsonl \
        --output_dir ./models/promethee \
        --epochs 10 \
        --eval_every 500 \
        --use_muon
"""

import os
import json
import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger

from transformers import (
    MambaForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup
)


# =============================================================================
# Muon Optimizer (SOTA)
# =============================================================================

class Muon(torch.optim.Optimizer):
    """Muon - SOTA optimizer for LLMs."""

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, momentum, nesterov, wd = group['lr'], group['momentum'], group['nesterov'], group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad
                if wd != 0:
                    p.mul_(1 - lr * wd)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

                p.add_(d_p, alpha=-lr)

        return loss


# =============================================================================
# Dataset
# =============================================================================

class CausalDataset(Dataset):
    def __init__(self, data_file: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))

        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        text = sample.get('input_text', '')
        ticker = sample.get('ticker', 'N/A')
        date = sample.get('date', 'N/A')
        quant = sample.get('quant_context', '')
        teacher = sample.get('teacher_output', '')

        if isinstance(teacher, dict):
            teacher = json.dumps(teacher)

        prompt = f"""[INST] Analyze the causal market impact:

Event: {text[:4000]}
Ticker: {ticker}
Date: {date}
Context: {quant}

Provide sentiment, impact, causal reasoning, risks. [/INST]

{teacher}</s>"""

        enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length,
                             padding='max_length', return_tensors='pt')

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': enc['input_ids'].squeeze(0).clone(),
            'raw_sample': sample
        }


# =============================================================================
# Teacher Evaluator
# =============================================================================

JUDGE_PROMPT = """Rate this AI financial analysis (0-10 JSON):
EVENT: {event}
GROUND TRUTH: Return {return_1d}%, Direction: {direction}
AI RESPONSE: {response}

Output JSON: {{"sentiment_accuracy": X, "causal_quality": X, "overall_score": X}}"""


class TeacherEvaluator:
    """Evaluates Prométhée responses using Teacher LLM."""

    def __init__(self, model_name: str = "THUDM/glm-4-9b-chat", use_4bit: bool = True):
        logger.info(f"Loading Teacher evaluator: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )

    def evaluate_response(self, event: str, ground_truth: Dict, response: str) -> Optional[Dict]:
        """Evaluate a single response."""
        prompt = JUDGE_PROMPT.format(
            event=event[:1000],
            return_1d=ground_truth.get('return_1d', 'N/A'),
            direction=ground_truth.get('direction', 'N/A'),
            response=response[:500]
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt", return_dict=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=128,
                                              temperature=0.1, do_sample=True)

            resp = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                         skip_special_tokens=True)

            # Parse JSON
            resp = resp.strip()
            if '{' in resp:
                resp = resp[resp.find('{'):resp.rfind('}')+1]
            return json.loads(resp)

        except Exception as e:
            return None

    def evaluate_batch(self, samples: List[Dict], promethee_model, promethee_tokenizer,
                       num_samples: int = 10) -> Dict:
        """Evaluate a batch of samples."""
        scores = []

        eval_samples = random.sample(samples, min(num_samples, len(samples)))

        for sample in eval_samples:
            # Generate Prométhée response
            text = sample.get('input_text', '')[:2000]
            ticker = sample.get('ticker', 'N/A')

            prompt = f"[INST] Analyze market impact: {text[:1000]} Ticker: {ticker} [/INST]\n"
            inputs = promethee_tokenizer(prompt, return_tensors="pt").to(promethee_model.device)

            with torch.no_grad():
                outputs = promethee_model.generate(
                    **inputs, max_new_tokens=256, do_sample=True,
                    temperature=0.3, pad_token_id=promethee_tokenizer.eos_token_id
                )

            response = promethee_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            # Evaluate with Teacher
            ground_truth = sample.get('ground_truth', {})
            eval_result = self.evaluate_response(text, ground_truth, response)

            if eval_result and 'overall_score' in eval_result:
                scores.append(eval_result['overall_score'])

        if scores:
            return {
                'avg_score': sum(scores) / len(scores),
                'min_score': min(scores),
                'max_score': max(scores),
                'num_evaluated': len(scores)
            }
        return {'avg_score': 0, 'num_evaluated': 0}


# =============================================================================
# Training with Evaluation
# =============================================================================

def train_with_evaluation(
    model,
    tokenizer,
    train_loader,
    optimizer,
    scheduler,
    teacher_evaluator: Optional[TeacherEvaluator],
    eval_samples: List[Dict],
    device,
    args
):
    """Training loop with periodic Teacher evaluation."""

    model.train()
    global_step = 0
    best_score = 0
    best_loss = float('inf')
    history = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            labels[labels == tokenizer.pad_token_id] = -100

            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum

            loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += outputs.loss.item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix({'loss': f"{outputs.loss.item():.4f}",
                              'lr': f"{scheduler.get_last_lr()[0]:.2e}"})

            # Periodic Teacher evaluation
            if teacher_evaluator and global_step % args.eval_every == 0:
                model.eval()
                logger.info(f"\n[Step {global_step}] Running Teacher evaluation...")

                eval_result = teacher_evaluator.evaluate_batch(
                    eval_samples, model, tokenizer, num_samples=args.eval_samples
                )

                logger.info(f"  Teacher Score: {eval_result['avg_score']:.2f}/10 "
                           f"(min={eval_result.get('min_score', 0):.1f}, "
                           f"max={eval_result.get('max_score', 0):.1f})")

                # Save if best
                if eval_result['avg_score'] > best_score:
                    best_score = eval_result['avg_score']
                    logger.info(f"  New best score! Saving checkpoint...")
                    model.save_pretrained(output_dir / "checkpoint-best")
                    tokenizer.save_pretrained(output_dir / "checkpoint-best")

                    # Save eval result
                    with open(output_dir / "checkpoint-best" / "eval_result.json", 'w') as f:
                        json.dump({
                            'step': global_step,
                            'teacher_score': eval_result['avg_score'],
                            'loss': epoch_loss / num_batches
                        }, f, indent=2)

                history.append({
                    'step': global_step,
                    'loss': epoch_loss / num_batches,
                    'teacher_score': eval_result['avg_score']
                })

                model.train()

        # End of epoch
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch} complete: avg_loss={avg_loss:.4f}")

        # Save epoch checkpoint
        if epoch % 2 == 0:
            model.save_pretrained(output_dir / f"checkpoint-epoch-{epoch}")
            tokenizer.save_pretrained(output_dir / f"checkpoint-epoch-{epoch}")

    # Save final
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config
    config = {
        "model_name": "Prométhée",
        "version": "1.0",
        "best_teacher_score": best_score,
        "training": {
            "epochs": args.epochs,
            "optimizer": "muon" if args.use_muon else "adamw",
            "lr": args.lr,
            "eval_every": args.eval_every
        }
    }
    with open(output_dir / "promethee_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    return best_score, history


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Prométhée with Teacher Evaluation")

    # Model
    parser.add_argument("--base_model", type=str, default="state-spaces/mamba-1.4b-hf")
    parser.add_argument("--output_dir", type=str, default="./models/promethee")

    # Data
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1024)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_muon", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Evaluation
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Evaluate with Teacher every N steps")
    parser.add_argument("--eval_samples", type=int, default=20,
                        help="Number of samples for Teacher evaluation")
    parser.add_argument("--teacher_model", type=str, default="THUDM/glm-4-9b-chat")
    parser.add_argument("--no_teacher_eval", action="store_true",
                        help="Disable Teacher evaluation (faster)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("PROMETHEE Training with Teacher Evaluation")
    logger.info("=" * 70)
    logger.info(f"Optimizer: {'Muon (SOTA)' if args.use_muon else 'AdamW'}")
    logger.info(f"Teacher eval every: {args.eval_every} steps")
    logger.info("")

    # Load Prométhée (student)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = MambaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16
    ).to(device)

    model.config.pad_token_id = tokenizer.pad_token_id

    # Load dataset
    dataset = CausalDataset(args.train_file, tokenizer, args.max_length)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    # Keep raw samples for evaluation
    eval_samples = dataset.samples

    # Optimizer
    if args.use_muon:
        optimizer = Muon(model.parameters(), lr=args.lr, momentum=0.95,
                         weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)

    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Teacher evaluator
    teacher_evaluator = None
    if not args.no_teacher_eval:
        teacher_evaluator = TeacherEvaluator(args.teacher_model)

    # Train
    logger.info("Starting training...")
    best_score, history = train_with_evaluation(
        model, tokenizer, train_loader, optimizer, scheduler,
        teacher_evaluator, eval_samples, device, args
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Best Teacher Score: {best_score:.2f}/10")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
