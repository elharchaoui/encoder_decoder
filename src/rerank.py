from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data import DenoisingDataset, TextPair, build_pairs_from_config
from src.evaluate import token_f1
from src.models import FrozenEncoderAutoregressiveDecoder, FrozenEncoderDecoderConfig
from src.train import resolve_device


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_model(cfg: dict, checkpoint: str, tokenizer, device: torch.device):
    model_cfg = cfg["model"]
    model = FrozenEncoderAutoregressiveDecoder(
        FrozenEncoderDecoderConfig(
            encoder_name=model_cfg["encoder_name"],
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.cls_token_id or tokenizer.bos_token_id,
            eos_token_id=tokenizer.sep_token_id or tokenizer.eos_token_id,
            decoder_layers=int(model_cfg["decoder_layers"]),
            decoder_heads=int(model_cfg["decoder_heads"]),
            decoder_ffn_dim=int(model_cfg["decoder_ffn_dim"]),
            dropout=float(model_cfg["dropout"]),
            init_decoder_embeddings_from_encoder=bool(
                model_cfg.get("init_decoder_embeddings_from_encoder", True)
            ),
            init_decoder_layers_from_encoder=bool(
                model_cfg.get("init_decoder_layers_from_encoder", False)
            ),
            tie_token_embeddings=bool(model_cfg.get("tie_token_embeddings", True)),
            use_cross_attention=bool(model_cfg.get("use_cross_attention", True)),
        )
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def candidate_loss(
    model,
    tokenizer,
    source: str,
    candidate: str,
    cfg: dict,
    device: torch.device,
) -> float:
    dataset = DenoisingDataset(
        [TextPair(source=source, target=candidate)],
        tokenizer=tokenizer,
        source_max_length=int(cfg["data"]["source_max_length"]),
        target_max_length=int(cfg["data"]["target_max_length"]),
    )
    batch = dataset[0]
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"].unsqueeze(0).to(device),
            attention_mask=batch["attention_mask"].unsqueeze(0).to(device),
            decoder_input_ids=batch["decoder_input_ids"].unsqueeze(0).to(device),
            labels=batch["labels"].unsqueeze(0).to(device),
        )
    return float(outputs["loss"].item())


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


def write_report(
    report_path: Path,
    config_path: str,
    checkpoint: str,
    metrics: dict[str, float],
    rows: list[dict[str, str]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reranking Evaluation Report",
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
                f"- Prediction: `{row['prediction']}`",
                f"- Gold rank: `{row['gold_rank']}`",
                f"- Gold loss: `{row['gold_loss']}`",
                f"- Best loss: `{row['best_loss']}`",
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
    parser.add_argument("--disable-cross-attention", action="store_true")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.disable_cross_attention:
        cfg["model"]["use_cross_attention"] = False
    seed = int(cfg["seed"])
    rng = random.Random(seed + 99)
    device = resolve_device(args.device or cfg["training"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    model = load_model(cfg, args.checkpoint, tokenizer, device)

    pairs = build_pairs_from_config(
        cfg=cfg,
        seed=seed,
        mask_token=tokenizer.mask_token or "[MASK]",
        split=args.split,
    )
    limit = min(args.limit, len(pairs))
    correct = 0
    reciprocal_rank_total = 0.0
    gold_loss_total = 0.0
    best_loss_total = 0.0
    gold_best_margin_total = 0.0
    prediction_f1_total = 0.0
    rows: list[dict[str, str]] = []

    for pair_index in tqdm(range(limit), desc="rerank"):
        pair = pairs[pair_index]
        candidates = build_candidates(
            pair_index=pair_index,
            pairs=pairs,
            rng=rng,
            num_distractors=args.num_distractors,
        )
        scored = [
            (
                candidate,
                candidate_loss(
                    model=model,
                    tokenizer=tokenizer,
                    source=pair.source,
                    candidate=candidate,
                    cfg=cfg,
                    device=device,
                ),
            )
            for candidate in candidates
        ]
        scored.sort(key=lambda item: item[1])
        prediction, best_loss = scored[0]
        gold_loss = next(loss for candidate, loss in scored if candidate == pair.target)
        gold_rank = next(index for index, (candidate, _) in enumerate(scored, start=1) if candidate == pair.target)
        correct += int(gold_rank == 1)
        reciprocal_rank_total += 1.0 / gold_rank
        gold_loss_total += gold_loss
        best_loss_total += best_loss
        gold_best_margin_total += gold_loss - best_loss
        prediction_f1_total += token_f1(prediction, pair.target)
        if len(rows) < 8:
            rows.append(
                {
                    "source": pair.source,
                    "gold": pair.target,
                    "prediction": prediction,
                    "gold_rank": str(gold_rank),
                    "gold_loss": f"{gold_loss:.4f}",
                    "best_loss": f"{best_loss:.4f}",
                }
            )

    metrics = {
        "examples": float(limit),
        "num_candidates": float(args.num_distractors + 1),
        "accuracy": correct / max(1, limit),
        "mean_reciprocal_rank": reciprocal_rank_total / max(1, limit),
        "gold_loss": gold_loss_total / max(1, limit),
        "best_loss": best_loss_total / max(1, limit),
        "gold_best_loss_margin": gold_best_margin_total / max(1, limit),
        "prediction_token_f1": prediction_f1_total / max(1, limit),
        "random_accuracy_baseline": 1.0 / (args.num_distractors + 1),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))

    report = args.report
    if report is None:
        checkpoint_name = Path(args.checkpoint).stem
        config_name = Path(args.config).stem
        report = f"reports/{config_name}_{checkpoint_name}_rerank_{args.split}.md"
    write_report(
        report_path=Path(report),
        config_path=args.config,
        checkpoint=args.checkpoint,
        metrics=metrics,
        rows=rows,
    )
    print(f"wrote_report={report}")


if __name__ == "__main__":
    main()
