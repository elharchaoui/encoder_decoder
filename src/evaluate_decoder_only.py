from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import build_pairs_from_config
from src.evaluate import exact_match, source_copy_ratio, token_f1
from src.train import resolve_device


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def write_decoder_only_report(
    report_path: Path,
    config_path: str,
    checkpoint: str,
    cfg: dict,
    metrics: dict[str, float],
    samples: list[dict[str, str]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Decoder-Only Evaluation Report",
        "",
        "## Run",
        "",
        f"- Config: `{config_path}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Model: `{cfg['model']['decoder_name']}`",
        f"- Data source: `{cfg['data']['source']}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| `{key}` | {value:.4f} |")
        else:
            lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Samples", ""])
    for idx, sample in enumerate(samples, start=1):
        lines.extend(
            [
                f"### Sample {idx}",
                "",
                f"- Source: `{sample['source']}`",
                f"- Target: `{sample['target']}`",
                f"- Prediction: `{sample['prediction']}`",
                f"- Exact match: `{sample['exact_match']}`",
                f"- Token F1: `{sample['token_f1']}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    prompt_text: str,
    source_max_length: int,
    max_new_tokens: int,
    min_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> str:
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=source_max_length,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        num_beams=num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = generated[0][input_ids.shape[1]:]
    return " ".join(tokenizer.decode(new_tokens, skip_special_tokens=True).strip().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default=None)
    parser.add_argument("--generation-limit", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg["training"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    if cfg["training"].get("precision") == "bf16":
        model = model.to(dtype=torch.bfloat16)
    model = model.to(device)
    model.eval()

    pairs = build_pairs_from_config(
        cfg=cfg, seed=int(cfg["seed"]), mask_token="[MASK]", split=args.split
    )
    prompt_prefix = cfg["model"].get("prompt_prefix", "")
    answer_prefix = cfg["model"].get("answer_prefix", " answer: ")
    source_max = int(cfg["data"]["source_max_length"])
    gen_cfg = cfg.get("generation", {})
    max_new_tokens = args.max_new_tokens or int(gen_cfg.get("max_new_tokens", 32))
    min_new_tokens = args.min_new_tokens or int(gen_cfg.get("min_new_tokens", 1))
    num_beams = args.num_beams or int(gen_cfg.get("num_beams", 1))
    num_samples = int(gen_cfg.get("num_samples", 8))
    limit = min(args.generation_limit, len(pairs))

    total_em = 0.0
    total_f1 = 0.0
    total_source_f1 = 0.0
    total_pred_source_copy = 0.0
    prediction_counts: Counter[str] = Counter()
    prediction_lengths: list[int] = []
    rows: list[dict[str, str]] = []

    for pair in tqdm(pairs[:limit], desc="generate"):
        prompt_text = f"{prompt_prefix}{pair.source}{answer_prefix}"
        prediction = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            source_max_length=source_max,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            device=device,
        )
        em = exact_match(prediction, pair.target)
        f1 = token_f1(prediction, pair.target)
        source_f1 = token_f1(pair.source, pair.target)
        pred_source_copy = source_copy_ratio(prediction, pair.source)
        total_em += em
        total_f1 += f1
        total_source_f1 += source_f1
        total_pred_source_copy += pred_source_copy
        prediction_counts[prediction] += 1
        prediction_lengths.append(len(prediction.split()))
        if len(rows) < num_samples:
            rows.append(
                {
                    "source": pair.source[:200],
                    "target": pair.target,
                    "prediction": prediction,
                    "exact_match": f"{em:.3f}",
                    "token_f1": f"{f1:.3f}",
                }
            )

    denom = max(1, limit)
    metrics = {
        "generation_examples": float(limit),
        "exact_match": total_em / denom,
        "token_f1": total_f1 / denom,
        "source_copy_token_f1": total_source_f1 / denom,
        "prediction_source_copy_ratio": total_pred_source_copy / denom,
        "unique_predictions": float(len(prediction_counts)),
        "top_prediction_ratio": (
            prediction_counts.most_common(1)[0][1] / denom if prediction_counts else 0.0
        ),
        "empty_prediction_ratio": prediction_counts.get("", 0) / denom,
        "avg_prediction_length_tokens": (
            sum(prediction_lengths) / max(1, len(prediction_lengths))
        ),
        "num_beams": float(num_beams),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))

    report = args.report
    if report is None:
        checkpoint_name = Path(args.checkpoint).name
        config_name = Path(args.config).stem
        report = f"reports/{config_name}_{checkpoint_name}_{args.split}.md"

    write_decoder_only_report(
        report_path=Path(report),
        config_path=args.config,
        checkpoint=args.checkpoint,
        cfg=cfg,
        metrics=metrics,
        samples=rows,
    )
    print(f"wrote_report={report}")


if __name__ == "__main__":
    main()
