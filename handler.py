"""
RunPod Serverless Handler for Promethee LLM Distillation

Supports three modes:
1. generate_labels - Generate teacher labels using GLM-4-9B
2. train_student - Train Mamba student model
3. inference - Run inference on trained model
"""

import runpod
import torch
import json
import os
from loguru import logger

# Global model cache
MODELS = {}


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


def load_student_model(model_path=None):
    """Load Mamba student model"""
    if "student" in MODELS and model_path is None:
        return MODELS["student"]

    from transformers import MambaForCausalLM, AutoTokenizer

    model_name = model_path or os.getenv("STUDENT_MODEL", "state-spaces/mamba-1.4b-hf")
    logger.info(f"Loading student model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = MambaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    MODELS["student"] = (model, tokenizer)
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


def train_student_model(training_data: list, config: dict = None) -> dict:
    """Train Mamba student model on teacher-generated labels"""
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset

    config = config or {}
    output_dir = config.get("output_dir", "/app/outputs/fin-mamba-student")

    model, tokenizer = load_student_model()

    # Format training data
    def format_example(sample):
        text = sample['input_text']
        target = sample['teacher_output']
        if isinstance(target, dict):
            target = json.dumps(target)
        prompt = f"Analyze the following financial text:\n\nInput: {text}\n\nAnalysis:\n{target}<|endoftext|>"
        return {"text": prompt}

    dataset = Dataset.from_list(training_data)
    dataset = dataset.map(format_example)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            truncation=True,
            max_length=config.get("max_length", 2048),
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 8),
        learning_rate=config.get("learning_rate", 2e-4),
        logging_steps=10,
        num_train_epochs=config.get("epochs", 3),
        save_strategy="steps",
        save_steps=500,
        fp16=True,
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return {"status": "success", "model_path": output_dir}


def run_inference(texts: list, model_path: str = None) -> list:
    """Run inference using trained student model"""
    model, tokenizer = load_student_model(model_path)

    results = []
    for text in texts:
        try:
            prompt = f"Analyze the following financial text:\n\nInput: {text[:5000]}\n\nAnalysis:\n"
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
            if "Analysis:" in response:
                response = response.split("Analysis:")[-1].strip()

            results.append({"status": "success", "output": response})

        except Exception as e:
            logger.error(f"Inference error: {e}")
            results.append({"status": "error", "error": str(e)})

    return results


def handler(job):
    """
    RunPod handler function

    Input format:
    {
        "mode": "generate_labels" | "train_student" | "inference",
        "texts": [...],  # For generate_labels or inference
        "training_data": [...],  # For train_student
        "config": {...},  # Optional configuration
        "model_path": "..."  # Optional for inference
    }
    """
    job_input = job["input"]
    mode = job_input.get("mode", "inference")

    logger.info(f"Processing job in mode: {mode}")

    try:
        if mode == "generate_labels":
            texts = job_input.get("texts", [])
            system_prompt = job_input.get("system_prompt")
            results = generate_teacher_labels(texts, system_prompt)
            return {"status": "success", "results": results}

        elif mode == "train_student":
            training_data = job_input.get("training_data", [])
            config = job_input.get("config", {})
            result = train_student_model(training_data, config)
            return result

        elif mode == "inference":
            texts = job_input.get("texts", [])
            model_path = job_input.get("model_path")
            results = run_inference(texts, model_path)
            return {"status": "success", "results": results}

        else:
            return {"status": "error", "error": f"Unknown mode: {mode}"}

    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    logger.info("Starting Promethee LLM RunPod handler...")
    runpod.serverless.start({"handler": handler})
