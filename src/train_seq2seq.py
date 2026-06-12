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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.data import build_pairs_from_config
from src.seq2seq import Seq2SeqBatchConfig, Seq2SeqDenoisingDataset
from src.train import move_batch, resolve_device


def _apply_lora(model, lora_cfg: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        target_modules=lora_cfg.get("target_modules", ["q", "v"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
    )
    return get_peft_model(model, config)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        losses.append(float(outputs.loss.item()))
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
    model_name = cfg["model"]["seq2seq_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    precision = cfg["training"].get("precision", "fp32")
    torch_dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision == "fp16" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch_dtype).to(device)
    if bool(cfg["training"].get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
    lora_cfg = cfg["model"].get("lora", {})
    if lora_cfg.get("enabled", False):
        model = _apply_lora(model, lora_cfg)
    else:
        if bool(cfg["model"].get("freeze_encoder", False)):
            for parameter in model.get_encoder().parameters():
                parameter.requires_grad = False
        if bool(cfg["model"].get("freeze_decoder", False)):
            for parameter in model.get_decoder().parameters():
                parameter.requires_grad = False
        train_only_patterns = cfg["model"].get("train_only_patterns", [])
        if train_only_patterns:
            for _, parameter in model.named_parameters():
                parameter.requires_grad = False
            for name, parameter in model.named_parameters():
                if any(pattern in name for pattern in train_only_patterns):
                    parameter.requires_grad = True
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = build_pairs_from_config(
        cfg=cfg,
        seed=seed,
        mask_token=tokenizer.mask_token or "<extra_id_0>",
        split="train",
    )
    val_pairs = build_pairs_from_config(
        cfg=cfg,
        seed=seed,
        mask_token=tokenizer.mask_token or "<extra_id_0>",
        split="validation",
    )
    batch_cfg = Seq2SeqBatchConfig(
        source_max_length=int(cfg["data"]["source_max_length"]),
        target_max_length=int(cfg["data"]["target_max_length"]),
    )
    prefix = cfg["model"].get("input_prefix", "")
    train_dataset = Seq2SeqDenoisingDataset(train_pairs, tokenizer, batch_cfg, prefix=prefix)
    val_dataset = Seq2SeqDenoisingDataset(val_pairs, tokenizer, batch_cfg, prefix=prefix)
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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}")
    print(f"seq2seq_name={model_name}")
    print(f"freeze_encoder={cfg['model'].get('freeze_encoder', False)}")
    print(f"freeze_decoder={cfg['model'].get('freeze_decoder', False)}")
    print(f"train_only_patterns={cfg['model'].get('train_only_patterns', [])}")
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

    use_lora = lora_cfg.get("enabled", False)

    def save_checkpoint(save_dir: Path) -> None:
        # torch.save() streams tensors one at a time (~peak = largest tensor ~128MB)
        # vs safetensors which buffers the full 1.4GB model → OOM on constrained RAM
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            torch.cuda.empty_cache()
            if use_lora:
                model.save_pretrained(save_dir)
            else:
                model.config.save_pretrained(save_dir)
                if hasattr(model, "generation_config"):
                    model.generation_config.save_pretrained(save_dir)
                torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
            tokenizer.save_pretrained(save_dir)
            print(f"checkpoint_saved={save_dir}")
        except Exception as exc:
            print(f"checkpoint_save_failed={save_dir} err={exc}")

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
                labels=batch["labels"],
            )
            loss = outputs.loss / accumulation_steps
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
                    save_checkpoint(output_dir / "best")
                    print(f"step={step} saved_best val_loss={best_val_loss:.4f}")

            if step % save_every == 0:
                save_checkpoint(output_dir / f"step_{step}")

            if step >= max_steps:
                break
    progress.close()
    save_checkpoint(output_dir / "final")
    if best_step:
        print(f"best_step={best_step} best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
