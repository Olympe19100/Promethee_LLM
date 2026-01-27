"""
Generate Teacher Logits for Knowledge Distillation

This script extracts logits from the teacher model (GLM-4-9B) for true
knowledge distillation. Since teacher and student have different vocabularies,
we save the full probability distribution and handle mapping during training.
"""

import json
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from loguru import logger
from pathlib import Path

# Default configuration
DEFAULT_INPUT_FILE = "data/sec_corpus_clean.jsonl"
DEFAULT_OUTPUT_DIR = "data/teacher_logits"
DEFAULT_MODEL = "THUDM/glm-4-9b-chat"
HF_DATASET = "Arnaud19/sec-10k-corpus"

SYSTEM_PROMPT = """You are a senior financial analyst. Your task is to analyze the provided extract from an SEC 10-K filing.
Output a valid JSON response with the following fields:
- sentiment: "Positive", "Negative", or "Neutral"
- key_risks: A list of 3-5 key risk factors identified.
- summary: A concise summary of the financial outlook or business status described.
- reasoning: An explanation of why you assigned the sentiment.
Ensure the output is pure JSON."""

# Temperature for soft targets (higher = softer distributions)
DISTILL_TEMPERATURE = 2.0


def load_model(model_name: str, use_4bit: bool = True):
    """Load teacher model"""
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )

    return model, tokenizer


def get_teacher_logits(model, tokenizer, text_chunk: str, max_context: int = 4096,
                       max_new_tokens: int = 512, temperature: float = DISTILL_TEMPERATURE):
    """
    Generate response AND extract logits from teacher model.

    Returns:
        dict with:
        - response_text: The generated text
        - response_tokens: Token IDs of the response
        - logits: Soft probability distribution at each position (top-k)
        - top_k_indices: Indices of top-k tokens at each position
    """
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this text:\n\n{text_chunk[:max_context]}"}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

        input_length = inputs['input_ids'].shape[1]

        # Generate with output scores
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,  # Low temp for generation
                top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Get generated tokens (excluding input)
        generated_tokens = outputs.sequences[0][input_length:].cpu()

        # Get logits at each generation step
        # scores is a tuple of (num_generated_tokens,) tensors of shape (batch, vocab)
        all_logits = []
        all_top_k_indices = []
        top_k = 100  # Save top 100 tokens per position (memory efficient)

        for step_scores in outputs.scores:
            # Apply temperature for softer distribution
            soft_logits = step_scores[0] / temperature
            probs = F.softmax(soft_logits, dim=-1)

            # Get top-k
            top_probs, top_indices = torch.topk(probs, k=top_k)

            all_logits.append(top_probs.cpu().numpy().astype(np.float16))
            all_top_k_indices.append(top_indices.cpu().numpy())

        # Decode response
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response_text = response_text.replace("```json", "").replace("```", "").strip()

        return {
            "response_text": response_text,
            "response_tokens": generated_tokens.tolist(),
            "top_k_probs": all_logits,  # List of (top_k,) arrays
            "top_k_indices": all_top_k_indices,  # List of (top_k,) arrays
            "vocab_size": model.config.vocab_size,
            "temperature": temperature
        }

    except Exception as e:
        logger.error(f"Error generating logits: {e}")
        return None


def download_from_huggingface(output_path: str = DEFAULT_INPUT_FILE):
    """Download dataset from HuggingFace Hub"""
    from datasets import load_dataset

    logger.info(f"Downloading dataset from HuggingFace: {HF_DATASET}")
    dataset = load_dataset(HF_DATASET, split="train")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    logger.info(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Downloaded {len(dataset)} documents")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate teacher logits for KD")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=1000)
    parser.add_argument("--from-hf", action="store_true")
    parser.add_argument("--temperature", type=float, default=DISTILL_TEMPERATURE)
    args = parser.parse_args()

    # Download if needed
    if args.from_hf or not os.path.exists(args.input):
        args.input = download_from_huggingface(args.input)

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Index file to track processed documents
    index_file = output_dir / "index.jsonl"
    processed = set()
    if index_file.exists():
        with open(index_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed.add(data.get('input_path'))
        logger.info(f"Resuming: {len(processed)} documents already processed")

    # Load model
    model, tokenizer = load_model(args.model, use_4bit=not args.no_4bit)

    # Save tokenizer info for later use
    tokenizer_info = {
        "model_name": args.model,
        "vocab_size": tokenizer.vocab_size,
        "temperature": args.temperature
    }
    with open(output_dir / "tokenizer_info.json", 'w') as f:
        json.dump(tokenizer_info, f)

    # Also save the tokenizer itself
    tokenizer.save_pretrained(output_dir / "teacher_tokenizer")

    # Process documents
    samples_processed = 0
    with open(args.input, 'r', encoding='utf-8') as infile, \
         open(index_file, 'a', encoding='utf-8') as idx_file:

        for line in tqdm(infile, desc="Generating logits"):
            if args.max_samples and samples_processed >= args.max_samples:
                break

            record = json.loads(line)
            file_path = record.get('path', record.get('id', str(samples_processed)))

            if file_path in processed:
                continue

            text = record['text']
            if len(text) < args.min_length:
                continue

            # Generate logits
            result = get_teacher_logits(
                model, tokenizer, text,
                temperature=args.temperature
            )

            if result and result.get("response_text"):
                try:
                    # Validate JSON response
                    json.loads(result["response_text"])

                    # Save logits to separate file (they're large)
                    sample_id = f"sample_{samples_processed:06d}"
                    logits_file = output_dir / f"{sample_id}.npz"

                    np.savez_compressed(
                        logits_file,
                        top_k_probs=np.array(result["top_k_probs"], dtype=np.float16),
                        top_k_indices=np.array(result["top_k_indices"], dtype=np.int32),
                        response_tokens=np.array(result["response_tokens"], dtype=np.int32)
                    )

                    # Save index entry
                    index_entry = {
                        "sample_id": sample_id,
                        "input_path": file_path,
                        "ticker": record.get('ticker', 'UNKNOWN'),
                        "input_text": text[:10000],
                        "response_text": result["response_text"],
                        "logits_file": str(logits_file.name),
                        "num_tokens": len(result["response_tokens"]),
                        "vocab_size": result["vocab_size"]
                    }
                    idx_file.write(json.dumps(index_entry) + "\n")
                    idx_file.flush()

                    samples_processed += 1

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON for {record.get('ticker')}")

    logger.info(f"Done. Processed {samples_processed} samples.")
    logger.info(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
