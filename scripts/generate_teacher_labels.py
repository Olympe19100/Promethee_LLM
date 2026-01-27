"""
Generate training labels using a Teacher LLM (GLM-4-9B)

This script processes SEC 10-K filings and generates structured analysis
using a quantized teacher model for student model distillation.
"""

import json
import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from loguru import logger

# Default configuration
DEFAULT_INPUT_FILE = "data/sec_corpus_clean.jsonl"
DEFAULT_OUTPUT_FILE = "data/sec_training_data.jsonl"
DEFAULT_MODEL = "THUDM/glm-4-9b-chat"

SYSTEM_PROMPT = """You are a senior financial analyst. Your task is to analyze the provided extract from an SEC 10-K filing.
Output a valid JSON response with the following fields:
- sentiment: "Positive", "Negative", or "Neutral"
- key_risks: A list of 3-5 key risk factors identified.
- summary: A concise summary of the financial outlook or business status described.
- reasoning: An explanation of why you assigned the sentiment.
Ensure the output is pure JSON."""


def load_model(model_name: str, use_4bit: bool = True):
    """Load teacher model with optional 4-bit quantization"""
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


def get_teacher_analysis(model, tokenizer, text_chunk: str, max_context: int = 10000) -> str:
    """Generate analysis for a text chunk"""
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

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.2,
                top_p=0.9
            )

        response = tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        response = response.replace("```json", "").replace("```", "").strip()

        return response

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate teacher labels for LLM distillation")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE, help="Input JSONL file")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE, help="Output JSONL file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Teacher model name")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--min-length", type=int, default=1000, help="Min text length to process")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        logger.info("Run process_sec_data.py first to create the corpus.")
        return

    model, tokenizer = load_model(args.model, use_4bit=not args.no_4bit)

    # Track already processed documents for resume capability
    processed_paths = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_paths.add(data.get('input_path'))
                except:
                    pass
        logger.info(f"Resuming: {len(processed_paths)} documents already processed")

    # Process documents
    samples_processed = 0
    with open(args.input, 'r', encoding='utf-8') as infile, \
         open(args.output, 'a', encoding='utf-8') as outfile:

        for line in tqdm(infile, desc="Generating labels"):
            if args.max_samples and samples_processed >= args.max_samples:
                break

            record = json.loads(line)
            file_path = record.get('path', record.get('id', str(samples_processed)))

            if file_path in processed_paths:
                continue

            text = record['text']
            if len(text) < args.min_length:
                continue

            analysis = get_teacher_analysis(model, tokenizer, text)

            if analysis:
                try:
                    # Validate JSON
                    json.loads(analysis)

                    training_example = {
                        "input_text": text[:10000],
                        "input_path": file_path,
                        "ticker": record.get('ticker', 'UNKNOWN'),
                        "teacher_output": analysis
                    }
                    outfile.write(json.dumps(training_example) + "\n")
                    outfile.flush()
                    samples_processed += 1

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON for {record.get('ticker', 'UNKNOWN')}")

    logger.info(f"Done. Processed {samples_processed} new samples.")
    logger.info(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
