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
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import build_pairs_from_config
from src.evaluate import exact_match, source_copy_ratio, target_length_bucket, token_f1
from src.prefix_memory_data import PrefixMemoryBatchConfig, PrefixMemoryDataset
from src.train import move_batch, resolve_device
from src.train_prefix_memory import ensure_pad_token, load_bridge, model_from_config


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_report(
    report_path: Path,
    config_path: str,
    checkpoint: str,
    cfg: dict,
    metrics: dict[str, float],
    samples: list[dict[str, str]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Prefix-Memory Evaluation Report",
        "",
        "## Run",
        "",
        f"- Config: `{config_path}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Encoder: `{cfg['model']['encoder_name']}`",
        f"- Decoder baseline: `{cfg['model']['decoder_name']}`",
        f"- Memory tokens: `{cfg['model'].get('memory_tokens', 64)}`",
        f"- Data objective: `{cfg['data'].get('objective', 'full_reconstruction')}`",
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
                f"- Prefix-memory prediction: `{sample['prefix_prediction']}`",
                f"- Decoder-only baseline: `{sample['decoder_baseline_prediction']}`",
                f"- Prefix-memory token F1: `{sample['prefix_token_f1']}`",
                f"- Decoder baseline token F1: `{sample['decoder_baseline_token_f1']}`",
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
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            target_input_ids=batch["target_input_ids"],
            target_attention_mask=batch["target_attention_mask"],
            labels=batch["labels"],
        )
        losses.append(float(outputs.loss.item()))
    return sum(losses) / max(1, len(losses))


def _clean_prediction(text: str) -> str:
    return " ".join(text.strip().split())


@torch.no_grad()
def generate_decoder_baseline(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    batch = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    generated = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    continuation = generated[0, batch["input_ids"].shape[1] :]
    return _clean_prediction(tokenizer.decode(continuation, skip_special_tokens=True))


@torch.no_grad()
def evaluate_generation(
    prefix_model,
    decoder_baseline,
    pairs,
    encoder_tokenizer,
    decoder_tokenizer,
    cfg: dict,
    device: torch.device,
    limit: int,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    prefix_source = cfg["model"].get("source_prefix", "")
    baseline_template = cfg["model"].get(
        "decoder_baseline_prompt",
        "Recover the missing span.\nText: {source}\nMissing span:",
    )
    max_new_tokens = int(cfg["generation"].get("max_new_tokens", 16))
    min_new_tokens = int(cfg["generation"].get("min_new_tokens", 0))
    rows: list[dict[str, str]] = []
    prefix_totals = {
        "exact_match": 0.0,
        "token_f1": 0.0,
        "source_copy_token_f1": 0.0,
        "prediction_source_copy_ratio": 0.0,
    }
    baseline_totals = {
        "exact_match": 0.0,
        "token_f1": 0.0,
        "prediction_source_copy_ratio": 0.0,
    }
    bucket_f1_totals: dict[str, float] = {}
    bucket_counts: dict[str, int] = {}
    prefix_prediction_counts: Counter[str] = Counter()
    baseline_prediction_counts: Counter[str] = Counter()

    for pair in tqdm(pairs[:limit], desc="generate", leave=False):
        source = encoder_tokenizer(
            f"{prefix_source}{pair.source}",
            max_length=int(cfg["data"]["source_max_length"]),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        decoder_prefix = cfg["model"].get("decoder_prefix", "")
        prompt = decoder_tokenizer(
            decoder_prefix,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prefix_ids = prefix_model.generate(
            source_input_ids=source["input_ids"].to(device),
            source_attention_mask=source["attention_mask"].to(device),
            bos_token_id=decoder_tokenizer.bos_token_id or decoder_tokenizer.eos_token_id,
            eos_token_id=decoder_tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            prompt_input_ids=prompt["input_ids"].to(device),
            prompt_attention_mask=prompt["attention_mask"].to(device),
        )
        prefix_prediction = _clean_prediction(
            decoder_tokenizer.decode(prefix_ids[0], skip_special_tokens=True)
        )
        baseline_prediction = generate_decoder_baseline(
            decoder_baseline,
            decoder_tokenizer,
            baseline_template.format(source=pair.source),
            device,
            max_new_tokens=max_new_tokens,
        )
        prefix_prediction_counts[prefix_prediction] += 1
        baseline_prediction_counts[baseline_prediction] += 1

        prefix_em = exact_match(prefix_prediction, pair.target)
        prefix_f1 = token_f1(prefix_prediction, pair.target)
        baseline_em = exact_match(baseline_prediction, pair.target)
        baseline_f1 = token_f1(baseline_prediction, pair.target)
        prefix_totals["exact_match"] += prefix_em
        prefix_totals["token_f1"] += prefix_f1
        prefix_totals["source_copy_token_f1"] += token_f1(pair.source, pair.target)
        prefix_totals["prediction_source_copy_ratio"] += source_copy_ratio(
            prefix_prediction,
            pair.source,
        )
        baseline_totals["exact_match"] += baseline_em
        baseline_totals["token_f1"] += baseline_f1
        baseline_totals["prediction_source_copy_ratio"] += source_copy_ratio(
            baseline_prediction,
            pair.source,
        )
        bucket = target_length_bucket(pair.target)
        bucket_f1_totals[bucket] = bucket_f1_totals.get(bucket, 0.0) + prefix_f1
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        if len(rows) < int(cfg["generation"].get("num_samples", 5)):
            rows.append(
                {
                    "source": pair.source,
                    "target": pair.target,
                    "prefix_prediction": prefix_prediction,
                    "decoder_baseline_prediction": baseline_prediction,
                    "prefix_token_f1": f"{prefix_f1:.3f}",
                    "decoder_baseline_token_f1": f"{baseline_f1:.3f}",
                }
            )

    denom = max(1, limit)
    metrics = {
        "generation_examples": float(limit),
        "prefix_exact_match": prefix_totals["exact_match"] / denom,
        "prefix_token_f1": prefix_totals["token_f1"] / denom,
        "source_copy_token_f1": prefix_totals["source_copy_token_f1"] / denom,
        "prefix_prediction_source_copy_ratio": (
            prefix_totals["prediction_source_copy_ratio"] / denom
        ),
        "decoder_baseline_exact_match": baseline_totals["exact_match"] / denom,
        "decoder_baseline_token_f1": baseline_totals["token_f1"] / denom,
        "decoder_baseline_prediction_source_copy_ratio": (
            baseline_totals["prediction_source_copy_ratio"] / denom
        ),
        "prefix_unique_predictions": float(len(prefix_prediction_counts)),
        "prefix_top_prediction_ratio": (
            prefix_prediction_counts.most_common(1)[0][1] / denom
            if prefix_prediction_counts
            else 0.0
        ),
        "decoder_baseline_unique_predictions": float(len(baseline_prediction_counts)),
        "decoder_baseline_top_prediction_ratio": (
            baseline_prediction_counts.most_common(1)[0][1] / denom
            if baseline_prediction_counts
            else 0.0
        ),
    }
    metrics["prefix_gain_over_source_copy"] = (
        metrics["prefix_token_f1"] - metrics["source_copy_token_f1"]
    )
    metrics["prefix_gap_to_decoder_baseline"] = (
        metrics["decoder_baseline_token_f1"] - metrics["prefix_token_f1"]
    )
    for bucket, total in sorted(bucket_f1_totals.items()):
        metrics[f"prefix_{bucket}_token_f1"] = total / max(1, bucket_counts[bucket])
        metrics[f"{bucket}_examples"] = float(bucket_counts[bucket])
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--generation-limit", type=int, default=64)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(args.device or cfg["training"]["device"])
    encoder_tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    decoder_tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["decoder_name"])
    ensure_pad_token(encoder_tokenizer)
    ensure_pad_token(decoder_tokenizer)

    prefix_model = model_from_config(cfg).to(device)
    if cfg["training"].get("precision") == "bf16":
        prefix_model = prefix_model.to(dtype=torch.bfloat16)
    load_bridge(prefix_model, args.checkpoint, device)
    prefix_model.eval()

    decoder_baseline = AutoModelForCausalLM.from_pretrained(cfg["model"]["decoder_name"]).to(device)
    if cfg["training"].get("precision") == "bf16":
        decoder_baseline = decoder_baseline.to(dtype=torch.bfloat16)
    decoder_baseline.eval()

    torch.manual_seed(int(cfg["seed"]))
    mask_token = cfg["data"].get("span_target", {}).get("sentinel_token", "[unused1]")
    pairs = build_pairs_from_config(
        cfg=cfg,
        seed=int(cfg["seed"]),
        mask_token=mask_token,
        split=args.split,
    )
    batch_cfg = PrefixMemoryBatchConfig(
        source_max_length=int(cfg["data"]["source_max_length"]),
        target_max_length=int(cfg["data"]["target_max_length"]),
    )
    dataset = PrefixMemoryDataset(
        pairs,
        encoder_tokenizer,
        decoder_tokenizer,
        batch_cfg,
        source_prefix=cfg["model"].get("source_prefix", ""),
        decoder_prefix=cfg["model"].get("decoder_prefix", ""),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
    )
    loss = evaluate_loss(prefix_model, loader, device)
    generation_limit = min(args.generation_limit, len(pairs))
    generation_metrics, samples = evaluate_generation(
        prefix_model=prefix_model,
        decoder_baseline=decoder_baseline,
        pairs=pairs,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        cfg=cfg,
        device=device,
        limit=generation_limit,
    )
    metrics = {
        "validation_loss" if args.split == "validation" else "train_loss": loss,
        "perplexity": math.exp(min(loss, 20)),
        **generation_metrics,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))

    report = args.report
    if report is None:
        checkpoint_name = Path(args.checkpoint).name
        config_name = Path(args.config).stem
        report = f"reports/{config_name}_{checkpoint_name}_{args.split}.md"
    write_report(
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
