"""
Evaluate Prométhée with Teacher LLM (LLM-as-a-Judge)

COMPREHENSIVE EVALUATION - Professional-grade financial analysis.

The Teacher (GLM-4-9B) evaluates if Prométhée provides:
1. Complete market analysis (not just sentiment)
2. Quantitative integration
3. Sector & macro context
4. Risk/reward assessment
5. Actionable insights
6. Time horizon considerations
7. Portfolio implications

Usage:
    python scripts/evaluate_with_teacher.py \
        --promethee_model ./models/promethee/checkpoint-best \
        --eval_file data/eval_samples.jsonl \
        --num_samples 100
"""

import os
import json
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from loguru import logger

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    MambaForCausalLM,
    BitsAndBytesConfig
)


# =============================================================================
# Professional-Grade Evaluation Prompt
# =============================================================================

JUDGE_SYSTEM_PROMPT = """You are a senior portfolio manager at a top-tier hedge fund.
You are evaluating an AI analyst's market analysis for professional trading decisions.

EVALUATION CRITERIA (0-10 scale each):

1. MARKET DYNAMICS (0-10):
   - Does the analysis understand supply/demand, price action drivers?
   - Are key market participants identified (institutions, retail, algos)?
   - Is liquidity/volume context considered?

2. FUNDAMENTAL ANALYSIS (0-10):
   - Are earnings, revenue, margins discussed if relevant?
   - Is valuation (P/E, EV/EBITDA, etc.) considered?
   - Are balance sheet risks identified?

3. SECTOR & MACRO CONTEXT (0-10):
   - How does this fit in the broader sector narrative?
   - Are macro factors (rates, inflation, GDP) integrated?
   - Is relative performance vs peers discussed?

4. QUANTITATIVE RIGOR (0-10):
   - Are specific numbers/percentages provided?
   - Is statistical context given (volatility, correlations)?
   - Are probabilities or confidence levels stated?

5. RISK ASSESSMENT (0-10):
   - Are downside scenarios clearly articulated?
   - Is tail risk considered?
   - Are hedging considerations mentioned?

6. CATALYST IDENTIFICATION (0-10):
   - Are specific upcoming catalysts identified (earnings, FDA, etc.)?
   - Is timing of expected move discussed?
   - Are second-order effects considered?

7. ACTIONABLE INSIGHTS (0-10):
   - Is there a clear trade recommendation?
   - Are entry/exit levels suggested?
   - Is position sizing or conviction level indicated?

8. PROFESSIONAL QUALITY (0-10):
   - Would a PM use this in an investment committee?
   - Is the reasoning sophisticated, not generic?
   - Is the writing clear and structured?

Output ONLY valid JSON:
{
  "market_dynamics": <0-10>,
  "fundamental_analysis": <0-10>,
  "sector_macro_context": <0-10>,
  "quantitative_rigor": <0-10>,
  "risk_assessment": <0-10>,
  "catalyst_identification": <0-10>,
  "actionable_insights": <0-10>,
  "professional_quality": <0-10>,
  "overall_score": <0-10>,
  "grade": "A/B/C/D/F",
  "would_trade_on_this": true/false,
  "key_strength": "<one sentence>",
  "key_weakness": "<one sentence>",
  "improvement_needed": "<specific suggestion>"
}"""


JUDGE_USER_TEMPLATE = """Evaluate this AI analyst's market analysis:

=== MARKET EVENT ===
{event_text}

Ticker: {ticker}
Date: {date}
Sector: {sector}

=== QUANTITATIVE CONTEXT ===
- Market Regime: {regime}
- VIX Level: {vix}
- Stock Momentum (20d): {momentum}
- Sector Performance (20d): {sector_perf}

=== ACTUAL MARKET OUTCOME ===
- 1-day return: {return_1d}%
- 5-day return: {return_5d}%
- Direction: {direction}
- Realized volatility: {volatility}%

=== AI ANALYST'S ANALYSIS ===
{model_response}

===
Rate this analysis as if you're deciding whether to use it for a $10M position.
Provide your evaluation as JSON:"""


# =============================================================================
# Professional Prompt for Prométhée
# =============================================================================

PROMETHEE_ANALYSIS_PROMPT = """[INST] You are Prométhée, a senior quantitative analyst at a top hedge fund.

Provide a COMPREHENSIVE professional market analysis:

=== EVENT ===
{event_text}

=== MARKET DATA ===
Ticker: {ticker}
Date: {date}
Sector: {sector}
Market Regime: {regime}
VIX: {vix}
20-day Momentum: {momentum}%
Sector Performance: {sector_perf}%

=== REQUIRED ANALYSIS ===

1. MARKET IMPACT ASSESSMENT
   - Expected direction and magnitude
   - Confidence level with reasoning
   - Time horizon for the move

2. FUNDAMENTAL IMPLICATIONS
   - Impact on earnings/revenue expectations
   - Valuation implications
   - Balance sheet considerations

3. SECTOR & MACRO CONTEXT
   - How this fits the broader sector narrative
   - Macro factors to consider
   - Relative positioning vs peers

4. RISK/REWARD PROFILE
   - Upside scenario and probability
   - Downside scenario and probability
   - Key risks to monitor
   - Tail risk considerations

5. CATALYST TIMELINE
   - Immediate catalysts (1-5 days)
   - Medium-term catalysts (1-4 weeks)
   - Key dates to watch

6. ACTIONABLE RECOMMENDATION
   - Clear directional view
   - Suggested position sizing (conviction level)
   - Entry/exit considerations
   - Hedging suggestions if applicable

Be specific with numbers. No generic statements. Write as if presenting to investment committee.
[/INST]

"""


# =============================================================================
# Model Loaders
# =============================================================================

def load_teacher_model(model_name: str = "THUDM/glm-4-9b-chat", use_4bit: bool = True):
    """Load Teacher model for evaluation."""
    logger.info(f"Loading Teacher model: {model_name}")

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


def load_promethee_model(model_path: str):
    """Load Prométhée model."""
    logger.info(f"Loading Prométhée model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = MambaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return model, tokenizer


# =============================================================================
# Generation Functions
# =============================================================================

def generate_promethee_response(
    model,
    tokenizer,
    sample: Dict,
    max_new_tokens: int = 1024
) -> str:
    """Generate Prométhée's comprehensive analysis."""

    # Extract all context
    event_text = sample.get('input_text', sample.get('text', ''))[:3000]
    ticker = sample.get('ticker', 'UNKNOWN')
    date = sample.get('date', 'N/A')

    quant = sample.get('quant_features', {})
    regime = quant.get('regime', 'Unknown')
    vix = quant.get('vix', 'N/A')
    momentum = quant.get('momentum', 0)
    if momentum:
        momentum = f"{momentum * 100:.1f}" if isinstance(momentum, float) else str(momentum)
    else:
        momentum = "N/A"

    sector = quant.get('sector', 'Unknown')
    sector_perf = "N/A"

    # Build professional prompt
    prompt = PROMETHEE_ANALYSIS_PROMPT.format(
        event_text=event_text,
        ticker=ticker,
        date=date,
        sector=sector,
        regime=regime,
        vix=vix if vix else "N/A",
        momentum=momentum,
        sector_perf=sector_perf
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def get_teacher_evaluation(
    model,
    tokenizer,
    sample: Dict,
    model_response: str
) -> Optional[Dict]:
    """Get comprehensive Teacher evaluation."""

    event_text = sample.get('input_text', '')[:2000]
    ticker = sample.get('ticker', 'UNKNOWN')
    date = sample.get('date', 'N/A')

    quant = sample.get('quant_features', {})
    ground_truth = sample.get('ground_truth', {})

    user_prompt = JUDGE_USER_TEMPLATE.format(
        event_text=event_text,
        ticker=ticker,
        date=date,
        sector=quant.get('sector', 'Unknown'),
        regime=quant.get('regime', 'Unknown'),
        vix=quant.get('vix', 'N/A'),
        momentum=f"{quant.get('momentum', 0) * 100:.1f}" if quant.get('momentum') else "N/A",
        sector_perf="N/A",
        return_1d=ground_truth.get('return_1d', 'N/A'),
        return_5d=ground_truth.get('return_5d', 'N/A'),
        direction=ground_truth.get('direction', 'N/A'),
        volatility=ground_truth.get('volatility', 'N/A'),
        model_response=model_response[:3000]
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
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
                max_new_tokens=512,
                do_sample=True,
                temperature=0.1,
                top_p=0.9
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Parse JSON
        response = response.strip()
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        if "{" in response:
            response = response[response.find("{"):response.rfind("}")+1]

        evaluation = json.loads(response)
        return evaluation

    except Exception as e:
        logger.warning(f"Evaluation parse error: {e}")
        return None


# =============================================================================
# Evaluation Loop
# =============================================================================

@dataclass
class EvaluationResult:
    sample_id: int
    ticker: str
    date: str
    ground_truth: Dict
    promethee_response: str
    teacher_evaluation: Optional[Dict]
    scores: Dict


def run_evaluation(
    promethee_model,
    promethee_tokenizer,
    teacher_model,
    teacher_tokenizer,
    eval_samples: List[Dict],
    max_samples: int = 100
) -> List[EvaluationResult]:
    """Run comprehensive evaluation."""

    results = []

    for i, sample in enumerate(tqdm(eval_samples[:max_samples], desc="Evaluating")):
        # Generate comprehensive analysis
        promethee_response = generate_promethee_response(
            promethee_model,
            promethee_tokenizer,
            sample
        )

        # Get Teacher evaluation
        teacher_eval = get_teacher_evaluation(
            teacher_model,
            teacher_tokenizer,
            sample,
            promethee_response
        )

        # Extract scores
        score_keys = [
            'market_dynamics', 'fundamental_analysis', 'sector_macro_context',
            'quantitative_rigor', 'risk_assessment', 'catalyst_identification',
            'actionable_insights', 'professional_quality', 'overall_score'
        ]

        scores = {k: 0 for k in score_keys}
        if teacher_eval:
            for key in score_keys:
                if key in teacher_eval:
                    try:
                        scores[key] = float(teacher_eval[key])
                    except:
                        pass

        result = EvaluationResult(
            sample_id=i,
            ticker=sample.get('ticker', 'UNKNOWN'),
            date=sample.get('date', 'N/A'),
            ground_truth=sample.get('ground_truth', {}),
            promethee_response=promethee_response,
            teacher_evaluation=teacher_eval,
            scores=scores
        )
        results.append(result)

        # Log progress
        if (i + 1) % 10 == 0:
            valid = [r for r in results if r.teacher_evaluation]
            if valid:
                avg = sum(r.scores['overall_score'] for r in valid) / len(valid)
                tradeable = sum(1 for r in valid if r.teacher_evaluation.get('would_trade_on_this', False))
                logger.info(f"Progress: {i+1}/{max_samples} | Avg: {avg:.1f}/10 | Tradeable: {tradeable}/{len(valid)}")

    return results


def compute_aggregate_metrics(results: List[EvaluationResult]) -> Dict:
    """Compute comprehensive metrics."""

    valid = [r for r in results if r.teacher_evaluation is not None]
    if not valid:
        return {"error": "No valid evaluations"}

    metrics = {}

    # Score averages
    score_keys = [
        'market_dynamics', 'fundamental_analysis', 'sector_macro_context',
        'quantitative_rigor', 'risk_assessment', 'catalyst_identification',
        'actionable_insights', 'professional_quality', 'overall_score'
    ]

    for key in score_keys:
        scores = [r.scores[key] for r in valid]
        metrics[f"avg_{key}"] = sum(scores) / len(scores)

    # Grade distribution
    grades = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    for r in valid:
        grade = r.teacher_evaluation.get('grade', 'F')
        if grade in grades:
            grades[grade] += 1
    metrics['grade_distribution'] = grades

    # Would trade ratio
    tradeable = sum(1 for r in valid if r.teacher_evaluation.get('would_trade_on_this', False))
    metrics['tradeable_ratio'] = tradeable / len(valid)

    # Direction accuracy
    direction_correct = 0
    direction_total = 0
    for r in valid:
        gt_dir = r.ground_truth.get('direction', '')
        resp = r.promethee_response.lower()

        if 'bullish' in resp or 'long' in resp or 'buy' in resp:
            pred = 'up'
        elif 'bearish' in resp or 'short' in resp or 'sell' in resp:
            pred = 'down'
        else:
            pred = 'neutral'

        if gt_dir:
            direction_total += 1
            if pred == gt_dir:
                direction_correct += 1

    if direction_total > 0:
        metrics['direction_accuracy'] = direction_correct / direction_total

    metrics['num_evaluated'] = len(valid)
    metrics['num_failed'] = len(results) - len(valid)

    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Prométhée Evaluation")
    parser.add_argument("--promethee_model", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, default="THUDM/glm-4-9b-chat")
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="evaluation_results.json")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PROMETHEE - Professional-Grade Evaluation")
    logger.info("=" * 70)
    logger.info(f"Prométhée: {args.promethee_model}")
    logger.info(f"Teacher: {args.teacher_model}")
    logger.info(f"Samples: {args.num_samples}")
    logger.info("")

    # Load models
    promethee_model, promethee_tokenizer = load_promethee_model(args.promethee_model)
    teacher_model, teacher_tokenizer = load_teacher_model(args.teacher_model, not args.no_4bit)

    # Load data
    eval_samples = []
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            eval_samples.append(json.loads(line))

    logger.info(f"Loaded {len(eval_samples)} samples")

    # Run evaluation
    results = run_evaluation(
        promethee_model, promethee_tokenizer,
        teacher_model, teacher_tokenizer,
        eval_samples, args.num_samples
    )

    # Compute metrics
    metrics = compute_aggregate_metrics(results)

    # Display
    logger.info("")
    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)

    logger.info(f"\nSamples: {metrics.get('num_evaluated', 0)} evaluated, {metrics.get('num_failed', 0)} failed")

    logger.info("\n--- PROFESSIONAL SCORES (0-10) ---")
    logger.info(f"Market Dynamics:      {metrics.get('avg_market_dynamics', 0):.1f}")
    logger.info(f"Fundamental Analysis: {metrics.get('avg_fundamental_analysis', 0):.1f}")
    logger.info(f"Sector/Macro Context: {metrics.get('avg_sector_macro_context', 0):.1f}")
    logger.info(f"Quantitative Rigor:   {metrics.get('avg_quantitative_rigor', 0):.1f}")
    logger.info(f"Risk Assessment:      {metrics.get('avg_risk_assessment', 0):.1f}")
    logger.info(f"Catalyst ID:          {metrics.get('avg_catalyst_identification', 0):.1f}")
    logger.info(f"Actionable Insights:  {metrics.get('avg_actionable_insights', 0):.1f}")
    logger.info(f"Professional Quality: {metrics.get('avg_professional_quality', 0):.1f}")
    logger.info(f"OVERALL SCORE:        {metrics.get('avg_overall_score', 0):.1f}")

    logger.info(f"\n--- TRADING METRICS ---")
    logger.info(f"Would Trade On This:  {metrics.get('tradeable_ratio', 0)*100:.1f}%")
    if 'direction_accuracy' in metrics:
        logger.info(f"Direction Accuracy:   {metrics['direction_accuracy']*100:.1f}%")

    logger.info(f"\n--- GRADE DISTRIBUTION ---")
    grades = metrics.get('grade_distribution', {})
    for g in ['A', 'B', 'C', 'D', 'F']:
        logger.info(f"  {g}: {grades.get(g, 0)}")

    # Quality assessment
    overall = metrics.get('avg_overall_score', 0)
    if overall >= 8:
        quality = "EXCELLENT - Production-ready for trading desk"
    elif overall >= 7:
        quality = "GOOD - Suitable for analyst support"
    elif overall >= 5:
        quality = "FAIR - Needs improvement for professional use"
    else:
        quality = "POOR - Major training needed"

    logger.info(f"\n>>> QUALITY: {quality}")

    # Save
    output = {
        "metrics": metrics,
        "config": vars(args),
        "sample_analyses": [
            {
                "ticker": r.ticker,
                "date": r.date,
                "ground_truth": r.ground_truth,
                "response_preview": r.promethee_response[:1000],
                "scores": r.scores,
                "grade": r.teacher_evaluation.get('grade') if r.teacher_evaluation else None,
                "would_trade": r.teacher_evaluation.get('would_trade_on_this') if r.teacher_evaluation else None,
                "feedback": {
                    "strength": r.teacher_evaluation.get('key_strength') if r.teacher_evaluation else None,
                    "weakness": r.teacher_evaluation.get('key_weakness') if r.teacher_evaluation else None,
                    "improvement": r.teacher_evaluation.get('improvement_needed') if r.teacher_evaluation else None
                }
            }
            for r in results[:30]
        ]
    }

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()
