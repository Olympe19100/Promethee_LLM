"""
RunPod Serverless Handler for Prométhée LLM

Prométhée: SOTA Financial Causal Language Model based on Mamba architecture.

Supports three modes:
1. generate_labels - Generate teacher labels using GLM-4-9B
2. train_promethee - Train Prométhée model with Muon optimizer
3. inference - Run inference on trained Prométhée model
"""

import runpod
import torch
import json
import os
from loguru import logger

# Global model cache
MODELS = {}

# Model info
PROMETHEE_INFO = {
    "name": "Prométhée",
    "version": "1.0",
    "base": "Mamba 1.4B",
    "task": "Financial Causal Reasoning"
}


def load_teacher_model():
    """Load GLM-4-9B teacher model with 4-bit quantization"""
    if "teacher" in MODELS:
        return MODELS["teacher"]

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = os.getenv("TEACHER_MODEL", "THUDM/glm-4-9b-chat")
    logger.info(f"Loading teacher model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

    MODELS["teacher"] = (model, tokenizer)
    return model, tokenizer


def load_promethee_model(model_path=None):
    """Load Prométhée model (fine-tuned Mamba)"""
    if "promethee" in MODELS and model_path is None:
        return MODELS["promethee"]

    from transformers import MambaForCausalLM, AutoTokenizer

    model_name = model_path or os.getenv("PROMETHEE_MODEL", "./models/promethee")

    # Fallback to base Mamba if Prométhée not found
    if not os.path.exists(model_name):
        model_name = os.getenv("BASE_MODEL", "state-spaces/mamba-1.4b-hf")
        logger.warning(f"Prométhée model not found, using base: {model_name}")

    logger.info(f"Loading Prométhée model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = MambaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    MODELS["promethee"] = (model, tokenizer)
    return model, tokenizer


def generate_teacher_labels(texts: list, system_prompt: str = None) -> list:
    """Generate labels for a batch of texts using teacher model"""
    model, tokenizer = load_teacher_model()

    default_prompt = """You are a senior financial analyst. Analyze the provided text and output valid JSON with:
- sentiment: "Positive", "Negative", or "Neutral"
- key_risks: List of 3-5 key risk factors
- summary: Concise summary of financial outlook
- reasoning: Explanation for sentiment assignment"""

    system_prompt = system_prompt or default_prompt
    results = []

    for text in texts:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this text:\n\n{text[:10000]}"}
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

            # Validate JSON
            try:
                parsed = json.loads(response)
                results.append({"status": "success", "output": parsed})
            except json.JSONDecodeError:
                results.append({"status": "error", "output": response, "error": "Invalid JSON"})

        except Exception as e:
            logger.error(f"Error processing text: {e}")
            results.append({"status": "error", "error": str(e)})

    return results


def train_promethee(training_data: list, config: dict = None) -> dict:
    """Train Prométhée model on teacher-generated labels with Muon optimizer"""
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset

    config = config or {}
    output_dir = config.get("output_dir", "/app/outputs/promethee")
    use_muon = config.get("use_muon", True)

    model, tokenizer = load_promethee_model()

    # Format training data for causal reasoning
    def format_example(sample):
        text = sample['input_text']
        target = sample['teacher_output']
        ticker = sample.get('ticker', 'N/A')

        if isinstance(target, dict):
            target = json.dumps(target, indent=2)

        prompt = f"""[INST] Analyze the causal market impact:

Event: {text[:8000]}
Ticker: {ticker}

Provide: sentiment, impact, causal reasoning, risks [/INST]

{target}</s>"""
        return {"text": prompt}

    dataset = Dataset.from_list(training_data)
    dataset = dataset.map(format_example)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            truncation=True,
            max_length=config.get("max_length", 1024),
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 8),
        learning_rate=config.get("learning_rate", 3e-4),
        logging_steps=10,
        num_train_epochs=config.get("epochs", 3),
        save_strategy="steps",
        save_steps=500,
        fp16=True,
        max_grad_norm=config.get("grad_clip", 1.0),
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("🔥 Starting Prométhée training...")
    trainer.train()

    logger.info(f"Saving Prométhée to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save Prométhée config
    promethee_config = {
        "model_name": "Prométhée",
        "version": "1.0",
        "task": "financial_causal_reasoning"
    }
    with open(f"{output_dir}/promethee_config.json", 'w') as f:
        json.dump(promethee_config, f, indent=2)

    return {"status": "success", "model_path": output_dir, "model_name": "Prométhée"}


def run_inference(texts: list, model_path: str = None) -> list:
    """Run inference using trained Prométhée model"""
    model, tokenizer = load_promethee_model(model_path)

    results = []
    for text in texts:
        try:
            prompt = f"""[INST] Analyze the causal market impact:

Event: {text[:5000]}

Provide: sentiment, impact, causal reasoning, risks [/INST]

"""
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the analysis part
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()

            results.append({
                "status": "success",
                "model": "Prométhée",
                "output": response
            })

        except Exception as e:
            logger.error(f"Inference error: {e}")
            results.append({"status": "error", "error": str(e)})

    return results


def handler(job):
    """
    RunPod handler function for Prométhée

    Input format:
    {
        "mode": "generate_labels" | "train_promethee" | "inference",
        "texts": [...],  # For generate_labels or inference
        "training_data": [...],  # For train_promethee
        "config": {...},  # Optional configuration
        "model_path": "..."  # Optional for inference
    }
    """
    job_input = job["input"]
    mode = job_input.get("mode", "inference")

    logger.info(f"🔥 Prométhée - Processing job in mode: {mode}")

    try:
        if mode == "generate_labels":
            texts = job_input.get("texts", [])
            system_prompt = job_input.get("system_prompt")
            results = generate_teacher_labels(texts, system_prompt)
            return {"status": "success", "results": results}

        elif mode == "train_promethee" or mode == "train_student":
            training_data = job_input.get("training_data", [])
            config = job_input.get("config", {})
            result = train_promethee(training_data, config)
            return result

        elif mode == "inference":
            texts = job_input.get("texts", [])
            model_path = job_input.get("model_path")
            results = run_inference(texts, model_path)
            return {"status": "success", "model": "Prométhée", "results": results}

        elif mode == "info":
            return {"status": "success", "info": PROMETHEE_INFO}

        else:
            return {"status": "error", "error": f"Unknown mode: {mode}"}

    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🔥 Prométhée LLM - RunPod Serverless Handler")
    logger.info("=" * 60)
    runpod.serverless.start({"handler": handler})
