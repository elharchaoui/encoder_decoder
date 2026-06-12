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
from transformers import AutoTokenizer

from src.corruption import normalize_text
from src.data import DenoisingDataset, build_pairs_from_config
from src.models import FrozenEncoderAutoregressiveDecoder, FrozenEncoderDecoderConfig


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    model_type: str = "roberta-large",
    device: str = "cpu",
) -> dict[str, float]:
    """Compute corpus-level BERTScore P/R/F1 with baseline rescaling."""
    if not predictions or not references:
        return {}
    from bert_score import score as bs_score
    print(f"computing BERTScore ({len(predictions)} examples, model={model_type}, device={device})...")
    try:
        P, R, F1 = bs_score(
            predictions,
            references,
            model_type=model_type,
            lang="en",
            device=device,
            verbose=False,
            rescale_with_baseline=True,
        )
    except Exception:
        P, R, F1 = bs_score(
            predictions,
            references,
            model_type=model_type,
            lang="en",
            device=device,
            verbose=False,
            rescale_with_baseline=False,
        )
    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda, but CUDA is not available.")
    return torch.device(requested)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(target_tokens)
    overlap_count = sum(overlap.values())
    if overlap_count == 0:
        return 0.0
    precision = overlap_count / len(pred_tokens)
    recall = overlap_count / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, target: str) -> float:
    return float(normalize_text(prediction) == normalize_text(target))


def source_copy_ratio(prediction: str, source: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    source_tokens = set(normalize_text(source).split())
    if not pred_tokens:
        return 0.0
    copied = sum(1 for token in pred_tokens if token in source_tokens)
    return copied / len(pred_tokens)


def target_length_bucket(target: str) -> str:
    length = len(normalize_text(target).split())
    if length <= 1:
        return "target_len_1"
    if length <= 3:
        return "target_len_2_3"
    if length <= 6:
        return "target_len_4_6"
    return "target_len_7_plus"


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


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: torch.device) -> float:
    losses: list[float] = []
    for batch in tqdm(loader, desc="loss", leave=False):
        batch = move_batch(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            labels=batch["labels"],
        )
        losses.append(float(outputs["loss"].item()))
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
) -> tuple[dict[str, float], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    model.encoder_call_count = 0
    total_em = 0.0
    total_f1 = 0.0
    total_source_em = 0.0
    total_source_f1 = 0.0
    total_source_target_f1 = 0.0
    total_prediction_source_f1 = 0.0
    total_prediction_source_copy_ratio = 0.0
    bucket_f1_totals: dict[str, float] = {}
    bucket_counts: dict[str, int] = {}
    generation_cfg = {**cfg.get("generation", {}), **(generation_overrides or {})}

    for pair in tqdm(pairs[:limit], desc="generate", leave=False):
        batch = tokenizer(
            pair.source,
            max_length=int(cfg["data"]["source_max_length"]),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        generated = model.generate(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 64)),
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
        prediction = tokenizer.decode(generated[0], skip_special_tokens=True)
        em = exact_match(prediction, pair.target)
        f1 = token_f1(prediction, pair.target)
        source_em = exact_match(pair.source, pair.target)
        source_f1 = token_f1(pair.source, pair.target)
        source_target_f1 = token_f1(pair.source, pair.target)
        prediction_source_f1 = token_f1(prediction, pair.source)
        prediction_source_ratio = source_copy_ratio(prediction, pair.source)
        total_em += em
        total_f1 += f1
        total_source_em += source_em
        total_source_f1 += source_f1
        total_source_target_f1 += source_target_f1
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
        "source_target_token_f1": total_source_target_f1 / denom,
        "prediction_source_token_f1": total_prediction_source_f1 / denom,
        "prediction_source_copy_ratio": total_prediction_source_copy_ratio / denom,
        "encoder_calls_per_generation": model.encoder_call_count / denom,
    }
    for bucket, total in sorted(bucket_f1_totals.items()):
        metrics[f"{bucket}_token_f1"] = total / max(1, bucket_counts[bucket])
        metrics[f"{bucket}_examples"] = float(bucket_counts[bucket])
    metrics["token_f1_gain_over_source_copy"] = (
        metrics["token_f1"] - metrics["source_copy_token_f1"]
    )
    return metrics, rows


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
        "# Experiment Evaluation Report",
        "",
        "## Run",
        "",
        f"- Config: `{config_path}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Data source: `{cfg['data']['source']}`",
        f"- Data objective: `{cfg['data'].get('objective', 'full_reconstruction')}`",
        f"- Encoder: `{cfg['model']['encoder_name']}`",
        f"- Decoder layers: `{cfg['model']['decoder_layers']}`",
        f"- Decoder heads: `{cfg['model']['decoder_heads']}`",
        f"- Encoder frozen: `true`",
        f"- Decoder embedding init from encoder: `{cfg['model'].get('init_decoder_embeddings_from_encoder', True)}`",
        f"- Decoder layer init from encoder: `{cfg['model'].get('init_decoder_layers_from_encoder', False)}`",
        f"- Token embeddings tied: `{cfg['model'].get('tie_token_embeddings', True)}`",
        f"- Cross-attention enabled: `{cfg['model'].get('use_cross_attention', True)}`",
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
    parser.add_argument("--disable-cross-attention", action="store_true")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.disable_cross_attention:
        cfg["model"]["use_cross_attention"] = False
    device = resolve_device(args.device or cfg["training"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    model = load_model(cfg, args.checkpoint, tokenizer, device)
    torch.manual_seed(int(cfg["seed"]))

    pairs = build_pairs_from_config(
        cfg=cfg,
        seed=int(cfg["seed"]),
        mask_token=tokenizer.mask_token or "[MASK]",
        split=args.split,
    )
    dataset = DenoisingDataset(
        pairs,
        tokenizer=tokenizer,
        source_max_length=int(cfg["data"]["source_max_length"]),
        target_max_length=int(cfg["data"]["target_max_length"]),
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
    generation_metrics, samples = evaluate_generation(
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
    for key, value in generation_overrides.items():
        metrics[f"decode_{key}"] = float(value) if isinstance(value, (int, float, bool)) else value
    print(json.dumps(metrics, indent=2, sort_keys=True))

    report = args.report
    if report is None:
        checkpoint_name = Path(args.checkpoint).stem
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
