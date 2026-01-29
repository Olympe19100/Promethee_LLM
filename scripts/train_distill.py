"""
Train Mamba Student Model via Knowledge Distillation

This script trains a Mamba 1.4B model on teacher-generated labels
from SEC 10-K financial document analysis.
"""

import os
import json
import argparse
import torch
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
DEFAULT_OUTPUT_DIR = "./models/fin-mamba-student"


def format_instruction(sample):
    """Format input/output pair for causal language modeling"""
    text = sample['input_text']
    target = sample['teacher_output']

    # Handle both string and dict teacher outputs
    if isinstance(target, dict):
        target = json.dumps(target)

    prompt = f"Analyze the following financial text and provide a structured assessment:\n\nInput: {text}\n\nAnalysis:\n{target}<|endoftext|>"
    return {"text": prompt}


def main():
    parser = argparse.ArgumentParser(description="Train Mamba student model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model name")
    parser.add_argument("--train-file", type=str, default=DEFAULT_TRAIN_FILE, help="Training data JSONL")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    logger.info(f"Loading Mamba model: {args.model}")

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
        report_to="tensorboard",
        logging_dir=f"{args.output_dir}/logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("Starting training...")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    logger.info(f"Saving model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
