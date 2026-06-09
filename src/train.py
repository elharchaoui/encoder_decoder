from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data import DenoisingDataset, build_pairs_from_config
from src.models import FrozenEncoderAutoregressiveDecoder, FrozenEncoderDecoderConfig


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requested device=cuda, but torch.cuda.is_available() is false. "
            "The local GPU is not available to this environment."
        )
    return torch.device(requested)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            labels=batch["labels"],
        )
        losses.append(float(outputs["loss"].item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(cfg["training"]["device"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = build_pairs_from_config(
        cfg=cfg,
        seed=seed,
        mask_token=tokenizer.mask_token or "[MASK]",
        split="train",
    )
    val_pairs = build_pairs_from_config(
        cfg=cfg,
        seed=seed,
        mask_token=tokenizer.mask_token or "[MASK]",
        split="validation",
    )
    train_dataset = DenoisingDataset(
        train_pairs,
        tokenizer=tokenizer,
        source_max_length=int(cfg["data"]["source_max_length"]),
        target_max_length=int(cfg["data"]["target_max_length"]),
    )
    val_dataset = DenoisingDataset(
        val_pairs,
        tokenizer=tokenizer,
        source_max_length=int(cfg["data"]["source_max_length"]),
        target_max_length=int(cfg["data"]["target_max_length"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
    )

    bos_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    eos_token_id = tokenizer.sep_token_id or tokenizer.eos_token_id
    if bos_token_id is None or eos_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer must provide BOS/CLS, EOS/SEP, and PAD token ids.")

    model = FrozenEncoderAutoregressiveDecoder(
        FrozenEncoderDecoderConfig(
            encoder_name=cfg["model"]["encoder_name"],
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_layers=int(cfg["model"]["decoder_layers"]),
            decoder_heads=int(cfg["model"]["decoder_heads"]),
            decoder_ffn_dim=int(cfg["model"]["decoder_ffn_dim"]),
            dropout=float(cfg["model"]["dropout"]),
            init_decoder_embeddings_from_encoder=bool(
                cfg["model"].get("init_decoder_embeddings_from_encoder", True)
            ),
            tie_token_embeddings=bool(cfg["model"].get("tie_token_embeddings", True)),
            use_cross_attention=bool(cfg["model"].get("use_cross_attention", True)),
        )
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}")
    print(f"use_cross_attention={model.config.use_cross_attention}")
    print(f"trainable_params={trainable_params:,}")
    print(f"total_params={total_params:,}")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    max_steps = int(cfg["training"]["max_steps"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg["training"]["warmup_steps"]),
        num_training_steps=max_steps,
    )
    accumulation_steps = int(cfg["training"]["gradient_accumulation_steps"])
    log_every = int(cfg["training"]["log_every"])
    eval_every = int(cfg["training"]["eval_every"])
    save_every = int(cfg["training"]["save_every"])

    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    running_loss = 0.0
    best_val_loss = float("inf")
    best_step = 0
    progress = tqdm(total=max_steps, desc="training")
    while step < max_steps:
        for batch in train_loader:
            batch = move_batch(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                labels=batch["labels"],
            )
            loss = outputs["loss"] / accumulation_steps
            loss.backward()
            running_loss += float(loss.item()) * accumulation_steps

            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(cfg["training"]["gradient_clip_norm"]),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            progress.update(1)

            if step % log_every == 0:
                avg = running_loss / log_every
                running_loss = 0.0
                print(f"step={step} train_loss={avg:.4f} ppl={math.exp(min(avg, 20)):.2f}")

            if step % eval_every == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"step={step} val_loss={val_loss:.4f} val_ppl={math.exp(min(val_loss, 20)):.2f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    torch.save(model.state_dict(), output_dir / "best.pt")
                    print(f"step={step} saved_best val_loss={best_val_loss:.4f}")

            if step % save_every == 0:
                torch.save(model.state_dict(), output_dir / f"step_{step}.pt")

            if step >= max_steps:
                break
    progress.close()
    torch.save(model.state_dict(), output_dir / "final.pt")
    if best_step:
        print(f"best_step={best_step} best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
