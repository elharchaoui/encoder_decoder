from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.data import build_pairs_from_config
from src.evaluate import (
    compute_bertscore,
    exact_match,
    source_copy_ratio,
    target_length_bucket,
    token_f1,
)
from src.seq2seq import Seq2SeqBatchConfig, Seq2SeqDenoisingDataset
from src.train import move_batch, resolve_device


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_seq2seq_report(
    report_path: Path,
    config_path: str,
    checkpoint: str,
    cfg: dict,
    metrics: dict[str, float],
    samples: list[dict[str, str]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Seq2Seq Upper-Bound Evaluation Report",
        "",
        "## Run",
        "",
        f"- Config: `{config_path}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Data source: `{cfg['data']['source']}`",
        f"- Data objective: `{cfg['data'].get('objective', 'full_reconstruction')}`",
        f"- Seq2Seq model: `{cfg['model']['seq2seq_name']}`",
        f"- Input prefix: `{cfg['model'].get('input_prefix', '')}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        if key.endswith("loss"):
            display = f"{value:.4f}"
        elif key.endswith("perplexity"):
            display = f"{value:.2f}"
        else:
            display = f"{value:.4f}"
        lines.append(f"| `{key}` | {display} |")

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
                f"- Prediction-source copy ratio: `{sample.get('prediction_source_copy_ratio', 'n/a')}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in tqdm(loader, desc="loss", leave=False):
        batch = move_batch(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        losses.append(float(outputs.loss.item()))
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def evaluate_generation(
    model,
    pairs,
    tokenizer,
    cfg: dict,
    device: torch.device,
    limit: int,
    generation_overrides: dict | None = None,
) -> tuple[dict[str, float], list[dict[str, str]], list[str], list[str]]:
    rows: list[dict[str, str]] = []
    all_predictions: list[str] = []
    all_references: list[str] = []
    prediction_counts: Counter[str] = Counter()
    prediction_lengths: list[int] = []
    total_em = 0.0
    total_f1 = 0.0
    total_source_em = 0.0
    total_source_f1 = 0.0
    total_prediction_source_f1 = 0.0
    total_prediction_source_copy_ratio = 0.0
    bucket_f1_totals: dict[str, float] = {}
    bucket_counts: dict[str, int] = {}
    generation_cfg = {**cfg.get("generation", {}), **(generation_overrides or {})}
    prefix = cfg["model"].get("input_prefix", "")

    for pair in tqdm(pairs[:limit], desc="generate", leave=False):
        batch = tokenizer(
            f"{prefix}{pair.source}",
            max_length=int(cfg["data"]["source_max_length"]),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        generated = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 16)),
            do_sample=bool(generation_cfg.get("do_sample", False)),
            temperature=float(generation_cfg.get("temperature", 1.0)),
            top_k=int(generation_cfg.get("top_k", 0)),
            top_p=float(generation_cfg.get("top_p", 1.0)),
            repetition_penalty=float(generation_cfg.get("repetition_penalty", 1.0)),
            no_repeat_ngram_size=int(generation_cfg.get("no_repeat_ngram_size", 0)),
            num_beams=int(generation_cfg.get("num_beams", 1)),
            length_penalty=float(generation_cfg.get("length_penalty", 1.0)),
            min_new_tokens=int(generation_cfg.get("min_new_tokens", 0)),
        )
        prediction = " ".join(tokenizer.decode(generated[0], skip_special_tokens=True).strip().split())
        all_predictions.append(prediction)
        all_references.append(pair.target)
        prediction_counts[prediction] += 1
        prediction_lengths.append(len(prediction.split()))
        em = exact_match(prediction, pair.target)
        f1 = token_f1(prediction, pair.target)
        source_em = exact_match(pair.source, pair.target)
        source_f1 = token_f1(pair.source, pair.target)
        prediction_source_f1 = token_f1(prediction, pair.source)
        prediction_source_ratio = source_copy_ratio(prediction, pair.source)
        total_em += em
        total_f1 += f1
        total_source_em += source_em
        total_source_f1 += source_f1
        total_prediction_source_f1 += prediction_source_f1
        total_prediction_source_copy_ratio += prediction_source_ratio
        bucket = target_length_bucket(pair.target)
        bucket_f1_totals[bucket] = bucket_f1_totals.get(bucket, 0.0) + f1
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if len(rows) < int(cfg["generation"].get("num_samples", 5)):
            rows.append(
                {
                    "source": pair.source,
                    "target": pair.target,
                    "prediction": prediction,
                    "exact_match": f"{em:.3f}",
                    "token_f1": f"{f1:.3f}",
                    "prediction_source_copy_ratio": f"{prediction_source_ratio:.3f}",
                }
            )

    denom = max(1, limit)
    metrics = {
        "generation_examples": float(limit),
        "exact_match": total_em / denom,
        "token_f1": total_f1 / denom,
        "source_copy_exact_match": total_source_em / denom,
        "source_copy_token_f1": total_source_f1 / denom,
        "prediction_source_token_f1": total_prediction_source_f1 / denom,
        "prediction_source_copy_ratio": total_prediction_source_copy_ratio / denom,
        "unique_predictions": float(len(prediction_counts)),
        "top_prediction_ratio": (
            prediction_counts.most_common(1)[0][1] / denom if prediction_counts else 0.0
        ),
        "empty_prediction_ratio": prediction_counts.get("", 0) / denom,
        "avg_prediction_length_tokens": (
            sum(prediction_lengths) / max(1, len(prediction_lengths))
        ),
        "max_prediction_length_tokens": float(max(prediction_lengths) if prediction_lengths else 0),
    }
    for bucket, total in sorted(bucket_f1_totals.items()):
        metrics[f"{bucket}_token_f1"] = total / max(1, bucket_counts[bucket])
        metrics[f"{bucket}_examples"] = float(bucket_counts[bucket])
    metrics["token_f1_gain_over_source_copy"] = (
        metrics["token_f1"] - metrics["source_copy_token_f1"]
    )
    return metrics, rows, all_predictions, all_references


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--generation-limit", type=int, default=64)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--min-new-tokens", type=int, default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--bertscore", action="store_true", help="Compute BERTScore after generation (adds ~2min)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg["training"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    adapter_cfg_path = Path(args.checkpoint) / "adapter_config.json"
    if adapter_cfg_path.exists():
        import json as _json
        from peft import PeftModel
        base_name = _json.loads(adapter_cfg_path.read_text())["base_model_name_or_path"]
        base = AutoModelForSeq2SeqLM.from_pretrained(base_name)
        model = PeftModel.from_pretrained(base, args.checkpoint)
        model = model.merge_and_unload()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint)
    model = model.to(device)
    model.eval()
    torch.manual_seed(int(cfg["seed"]))

    pairs = build_pairs_from_config(
        cfg=cfg,
        seed=int(cfg["seed"]),
        mask_token=tokenizer.mask_token or "<extra_id_0>",
        split=args.split,
    )
    batch_cfg = Seq2SeqBatchConfig(
        source_max_length=int(cfg["data"]["source_max_length"]),
        target_max_length=int(cfg["data"]["target_max_length"]),
    )
    dataset = Seq2SeqDenoisingDataset(
        pairs,
        tokenizer,
        batch_cfg,
        prefix=cfg["model"].get("input_prefix", ""),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
    )
    loss = evaluate_loss(model, loader, device)
    generation_limit = min(args.generation_limit, len(pairs))
    generation_overrides = {
        key: value
        for key, value in {
            "do_sample": args.do_sample if args.do_sample else None,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "min_new_tokens": args.min_new_tokens,
        }.items()
        if value is not None
    }
    if args.do_sample and args.num_beams is None:
        generation_overrides["num_beams"] = 1
    generation_metrics, samples, all_predictions, all_references = evaluate_generation(
        model=model,
        pairs=pairs,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        limit=generation_limit,
        generation_overrides=generation_overrides,
    )
    metrics = {
        "validation_loss" if args.split == "validation" else "train_loss": loss,
        "perplexity": math.exp(min(loss, 20)),
        **generation_metrics,
    }
    if args.bertscore:
        metrics.update(compute_bertscore(all_predictions, all_references))
    for key, value in generation_overrides.items():
        metrics[f"decode_{key}"] = float(value) if isinstance(value, (int, float, bool)) else value
    print(json.dumps(metrics, indent=2, sort_keys=True))

    report = args.report
    if report is None:
        checkpoint_name = Path(args.checkpoint).name
        config_name = Path(args.config).stem
        report = f"reports/{config_name}_{checkpoint_name}_{args.split}.md"
    write_seq2seq_report(
        report_path=Path(report),
        config_path=args.config,
        checkpoint=args.checkpoint,
        cfg=cfg,
        metrics=metrics,
        samples=samples,
    )
    print(f"wrote_report={report}")


if __name__ == "__main__":
    main()
