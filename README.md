# Promethee LLM

Financial LLM Distillation Pipeline - Train a lightweight Mamba model on SEC 10-K analysis using knowledge distillation from GLM-4-9B.

## Architecture

```
GLM-4-9B (Teacher)  -->  Knowledge Distillation  -->  Mamba 1.4B (Student)
     |                                                      |
  9B params                                             1.4B params
  4-bit quant                                           Fast inference
```

## Quick Start

### Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download SEC filings
python scripts/download_sec_10k.py --tickers AAPL MSFT GOOGL --amount 5

# Process into clean text
python scripts/process_sec_data.py

# Generate teacher labels (requires GPU with 16GB+ VRAM)
python scripts/generate_teacher_labels.py

# Train student model
python scripts/train_distill.py --epochs 3
```

### RunPod Deployment

```bash
# Build Docker image
docker build -t promethee-llm .

# Push to Docker Hub
docker tag promethee-llm your-username/promethee-llm
docker push your-username/promethee-llm

# Deploy on RunPod using the image
```

## API Usage (RunPod Serverless)

### Generate Teacher Labels

```json
{
  "input": {
    "mode": "generate_labels",
    "texts": ["SEC 10-K filing text here..."],
    "system_prompt": "Optional custom prompt"
  }
}
```

### Train Student Model

```json
{
  "input": {
    "mode": "train_student",
    "training_data": [
      {"input_text": "...", "teacher_output": "..."}
    ],
    "config": {
      "epochs": 3,
      "batch_size": 1,
      "learning_rate": 2e-4
    }
  }
}
```

### Inference

```json
{
  "input": {
    "mode": "inference",
    "texts": ["Financial text to analyze..."],
    "model_path": "/app/models/fin-mamba-student"
  }
}
```

## Project Structure

```
Promethee_LLM/
├── handler.py              # RunPod serverless handler
├── Dockerfile              # Container image
├── requirements.txt        # Dependencies
├── scripts/
│   ├── download_sec_10k.py      # Download SEC filings
│   ├── process_sec_data.py      # Clean and prepare data
│   ├── generate_teacher_labels.py  # Teacher inference
│   └── train_distill.py         # Student training
├── data/
│   ├── sec-edgar-filings/       # Raw SEC data
│   ├── sec_corpus_clean.jsonl   # Processed corpus
│   └── sec_training_data.jsonl  # Teacher labels
└── models/
    └── fin-mamba-student/       # Trained model
```

## Models

| Model | Role | Size | VRAM (4-bit) |
|-------|------|------|--------------|
| GLM-4-9B | Teacher | 9B | ~6GB |
| Mamba 1.4B | Student | 1.4B | ~3GB |

## Hardware Requirements

- **Teacher (label generation)**: GPU with 16GB+ VRAM (or 8GB with 4-bit)
- **Student (training)**: GPU with 8GB+ VRAM
- **Inference**: GPU with 4GB+ VRAM

## License

MIT
