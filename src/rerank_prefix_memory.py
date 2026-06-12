from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import TextPair, build_pairs_from_config
from src.evaluate import token_f1
from src.prefix_memory_data import PrefixMemoryBatchConfig, PrefixMemoryDataset
from src.train import move_batch, resolve_device
from src.train_prefix_memory import ensure_pad_token, load_bridge, model_from_config


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_candidates(
    pair_index: int,
    pairs: list[TextPair],
    rng: random.Random,
    num_distractors: int,
) -> list[str]:
    gold = pairs[pair_index].target
    pool = [pair.target for idx, pair in enumerate(pairs) if idx != pair_index and pair.target != gold]
    rng.shuffle(pool)
    candidates = [gold, *pool[:num_distractors]]
    rng.shuffle(candidates)
    return candidates


@torch.no_grad()
def prefix_candidate_loss(
    model,
    encoder_tokenizer,
    decoder_tokenizer,
    cfg: dict,
    source: str,
    candidate: str,
    device: torch.device,
) -> float:
    dataset = PrefixMemoryDataset(
        [TextPair(source=source, target=candidate)],
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        config=PrefixMemoryBatchConfig(
            source_max_length=int(cfg["data"]["source_max_length"]),
            target_max_length=int(cfg["data"]["target_max_length"]),
        ),
        source_prefix=cfg["model"].get("source_prefix", ""),
        decoder_prefix=cfg["model"].get("decoder_prefix", ""),
    )
    batch = move_batch(dataset[0], device)
    outputs = model(
        source_input_ids=batch["source_input_ids"].unsqueeze(0),
        source_attention_mask=batch["source_attention_mask"].unsqueeze(0),
        target_input_ids=batch["target_input_ids"].unsqueeze(0),
        target_attention_mask=batch["target_attention_mask"].unsqueeze(0),
        labels=batch["labels"].unsqueeze(0),
    )
    return float(outputs.loss.item())


@torch.no_grad()
def decoder_baseline_candidate_loss(
    model,
    tokenizer,
    cfg: dict,
    source: str,
    candidate: str,
    device: torch.device,
) -> float:
    prompt = cfg["model"].get(
        "decoder_baseline_prompt",
        "Answer the question using only the context.\n{source}\nAnswer:",
    ).format(source=source)
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        max_length=int(cfg["data"]["source_max_length"]),
        truncation=True,
    )["input_ids"]
    candidate_text = candidate
    if tokenizer.eos_token:
        candidate_text = f"{candidate_text}{tokenizer.eos_token}"
    candidate_ids = tokenizer(
        candidate_text,
        add_special_tokens=False,
        max_length=int(cfg["data"]["target_max_length"]),
        truncation=True,
    )["input_ids"]
    input_ids = torch.tensor([prompt_ids + candidate_ids], dtype=torch.long, device=device)
    labels = torch.tensor(
        [[-100] * len(prompt_ids) + candidate_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return float(outputs.loss.item())


def summarize_rankings(
    rankings: list[dict[str, float | int | str]],
    num_candidates: int,
    prefix: str,
) -> dict[str, float]:
    examples = len(rankings)
    correct = sum(1 for row in rankings if int(row[f"{prefix}_gold_rank"]) == 1)
    reciprocal = sum(1.0 / int(row[f"{prefix}_gold_rank"]) for row in rankings)
    gold_loss = sum(float(row[f"{prefix}_gold_loss"]) for row in rankings)
    best_loss = sum(float(row[f"{prefix}_best_loss"]) for row in rankings)
    prediction_f1 = sum(float(row[f"{prefix}_prediction_f1"]) for row in rankings)
    return {
        f"{prefix}_accuracy": correct / max(1, examples),
        f"{prefix}_mean_reciprocal_rank": reciprocal / max(1, examples),
        f"{prefix}_gold_loss": gold_loss / max(1, examples),
        f"{prefix}_best_loss": best_loss / max(1, examples),
        f"{prefix}_gold_best_loss_margin": (gold_loss - best_loss) / max(1, examples),
        f"{prefix}_prediction_token_f1": prediction_f1 / max(1, examples),
        "random_accuracy_baseline": 1.0 / num_candidates,
    }


def write_report(
    report_path: Path,
    config_path: str,
    checkpoint: str,
    metrics: dict[str, float],
    rows: list[dict[str, str]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Prefix-Memory Reranking Report",
        "",
        "## Run",
        "",
        f"- Config: `{config_path}`",
        f"- Checkpoint: `{checkpoint}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        lines.append(f"| `{key}` | {value:.4f} |")
    lines.extend(["", "## Samples", ""])
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"### Sample {idx}",
                "",
                f"- Source: `{row['source']}`",
                f"- Gold: `{row['gold']}`",
                f"- Prefix prediction: `{row['prefix_prediction']}`",
                f"- Prefix gold rank: `{row['prefix_gold_rank']}`",
                f"- Decoder baseline prediction: `{row['decoder_baseline_prediction']}`",
                f"- Decoder baseline gold rank: `{row['decoder_baseline_gold_rank']}`",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--num-distractors", type=int, default=7)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    rng = random.Random(seed + 131)
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

    pairs = build_pairs_from_config(
        cfg=cfg,
        seed=seed,
        mask_token=cfg["data"].get("span_target", {}).get("sentinel_token", "[unused1]"),
        split=args.split,
    )
    limit = min(args.limit, len(pairs))
    num_candidates = args.num_distractors + 1
    ranking_rows: list[dict[str, float | int | str]] = []
    sample_rows: list[dict[str, str]] = []

    for pair_index in tqdm(range(limit), desc="rerank"):
        pair = pairs[pair_index]
        candidates = build_candidates(pair_index, pairs, rng, args.num_distractors)

        prefix_scored = [
            (
                candidate,
                prefix_candidate_loss(
                    prefix_model,
                    encoder_tokenizer,
                    decoder_tokenizer,
                    cfg,
                    pair.source,
                    candidate,
                    device,
                ),
            )
            for candidate in candidates
        ]
        baseline_scored = [
            (
                candidate,
                decoder_baseline_candidate_loss(
                    decoder_baseline,
                    decoder_tokenizer,
                    cfg,
                    pair.source,
                    candidate,
                    device,
                ),
            )
            for candidate in candidates
        ]
        prefix_scored.sort(key=lambda item: item[1])
        baseline_scored.sort(key=lambda item: item[1])
        prefix_prediction, prefix_best_loss = prefix_scored[0]
        baseline_prediction, baseline_best_loss = baseline_scored[0]
        prefix_gold_loss = next(loss for candidate, loss in prefix_scored if candidate == pair.target)
        baseline_gold_loss = next(loss for candidate, loss in baseline_scored if candidate == pair.target)
        prefix_gold_rank = next(
            index for index, (candidate, _) in enumerate(prefix_scored, start=1) if candidate == pair.target
        )
        baseline_gold_rank = next(
            index for index, (candidate, _) in enumerate(baseline_scored, start=1) if candidate == pair.target
        )
        ranking_rows.append(
            {
                "prefix_gold_rank": prefix_gold_rank,
                "prefix_gold_loss": prefix_gold_loss,
                "prefix_best_loss": prefix_best_loss,
                "prefix_prediction_f1": token_f1(prefix_prediction, pair.target),
                "decoder_baseline_gold_rank": baseline_gold_rank,
                "decoder_baseline_gold_loss": baseline_gold_loss,
                "decoder_baseline_best_loss": baseline_best_loss,
                "decoder_baseline_prediction_f1": token_f1(baseline_prediction, pair.target),
            }
        )
        if len(sample_rows) < 8:
            sample_rows.append(
                {
                    "source": pair.source,
                    "gold": pair.target,
                    "prefix_prediction": prefix_prediction,
                    "prefix_gold_rank": str(prefix_gold_rank),
                    "decoder_baseline_prediction": baseline_prediction,
                    "decoder_baseline_gold_rank": str(baseline_gold_rank),
                }
            )

    metrics = {
        "examples": float(limit),
        "num_candidates": float(num_candidates),
        **summarize_rankings(ranking_rows, num_candidates, "prefix"),
        **summarize_rankings(ranking_rows, num_candidates, "decoder_baseline"),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))

    report = args.report
    if report is None:
        checkpoint_name = Path(args.checkpoint).name
        config_name = Path(args.config).stem
        report = f"reports/{config_name}_{checkpoint_name}_rerank_{args.split}.md"
    write_report(Path(report), args.config, args.checkpoint, metrics, sample_rows)
    print(f"wrote_report={report}")


if __name__ == "__main__":
    main()
