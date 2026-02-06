"""
Train Prométhée - Financial Causal Language Model

Prométhée is a Mamba 1.4B model fine-tuned for financial causal reasoning
via knowledge distillation from GLM-4-9B Teacher.

Features:
- Muon optimizer (SOTA) with gradient clipping
- Causal reasoning task framing
- Knowledge distillation from Teacher LLM
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    MambaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from loguru import logger

# Default configuration
DEFAULT_MODEL = "state-spaces/mamba-1.4b-hf"
DEFAULT_TRAIN_FILE = "data/sec_training_data.jsonl"
DEFAULT_OUTPUT_DIR = "./models/promethee"


# =============================================================================
# Muon Optimizer (SOTA for LLMs)
# =============================================================================

class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum-based optimizer without exponential moving averages.
    SOTA for LLM training, better than AdamW.
    """

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
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if wd != 0:
                    p.mul_(1 - lr * wd)

                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)

                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

                p.add_(d_p, alpha=-lr)

        return loss


def format_instruction(sample):
    """Format input/output pair for causal financial reasoning"""
    text = sample['input_text']
    target = sample['teacher_output']
    ticker = sample.get('ticker', 'N/A')
    date = sample.get('date', 'N/A')

    # Handle both string and dict teacher outputs
    if isinstance(target, dict):
        target = json.dumps(target, indent=2)

    prompt = f"""[INST] Analyze the causal market impact of this financial event:

Event: {text[:8000]}
Ticker: {ticker}
Date: {date}

Provide:
1. Sentiment (bullish/bearish/neutral)
2. Expected impact magnitude
3. Causal reasoning chain
4. Risk factors [/INST]

{target}</s>"""
    return {"text": prompt}


def main():
    parser = argparse.ArgumentParser(description="Train Prométhée - Financial Causal LLM")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model name")
    parser.add_argument("--train-file", type=str, default=DEFAULT_TRAIN_FILE, help="Training data JSONL")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--use-muon", action="store_true", help="Use Muon optimizer (SOTA)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🔥 Prométhée - Financial Causal LLM Training")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.model}")
    logger.info(f"Optimizer: {'Muon (SOTA)' if args.use_muon else 'AdamW'}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Gradient clipping: {args.grad_clip}")
    logger.info("")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # Check Mamba fast kernels availability
    try:
        import mamba_ssm  # noqa: F401
        logger.info("Mamba fast kernels: AVAILABLE")
    except ImportError:
        logger.warning(
            "Mamba fast kernels NOT installed — falling back to slow path (high VRAM usage). "
            "Install with: pip install causal-conv1d>=1.2.0 mamba-ssm>=1.2.0"
        )

    model = MambaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Enable gradient checkpointing to reduce VRAM usage
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing: enabled")

    # Load dataset
    if not os.path.exists(args.train_file):
        logger.error(f"Training file not found: {args.train_file}")
        logger.info("Run generate_teacher_labels.py first.")
        return

    logger.info(f"Loading dataset from: {args.train_file}")
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    logger.info(f"Dataset size: {len(dataset)} examples")

    dataset = dataset.map(format_instruction)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    # Training configuration
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        num_train_epochs=args.epochs,
        save_strategy="steps",
        save_steps=500,
        bf16=True,
        optim="adamw_torch",
        max_grad_norm=args.grad_clip,
        report_to="tensorboard",
        logging_dir=f"{args.output_dir}/logs"
    )

    # Custom optimizer if using Muon
    if args.use_muon:
        logger.info("Using Muon optimizer (SOTA)")
        optimizer = Muon(model.parameters(), lr=args.lr, momentum=0.95, weight_decay=0.01)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            optimizers=(optimizer, None)  # Custom optimizer, default scheduler
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

    logger.info("Starting Prométhée training...")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    logger.info(f"Saving Prométhée model to: {args.output_dir}")
    # Ensure weight tying flag is saved for proper model reload
    model.config.tie_word_embeddings = True
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save Prométhée config
    promethee_config = {
        "model_name": "Prométhée",
        "model_type": "mamba-causal-lm",
        "version": "1.0",
        "base_model": args.model,
        "task": "financial_causal_reasoning",
        "optimizer": "muon" if args.use_muon else "adamw",
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "gradient_clip": args.grad_clip
        }
    }
    with open(f"{args.output_dir}/promethee_config.json", 'w') as f:
        json.dump(promethee_config, f, indent=2)

    logger.info("")
    logger.info("🔥 Prométhée training complete!")


if __name__ == "__main__":
    main()
