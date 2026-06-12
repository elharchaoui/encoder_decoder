from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.data import TextPair, build_pairs_from_config
from src.train import resolve_device


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


class CausalQADataset(Dataset):
    """Causal LM fine-tuning dataset for QA.

    Input  : {prompt_prefix}{source}{answer_prefix}
    Full   : {prompt_prefix}{source}{answer_prefix}{target}{eos}
    Labels : -100 for prompt tokens, target+eos token ids for the answer.
    """

    def __init__(
        self,
        pairs: list[TextPair],
        tokenizer,
        source_max_length: int,
        target_max_length: int,
        prompt_prefix: str = "",
        answer_prefix: str = " answer: ",
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length
        self.prompt_prefix = prompt_prefix
        self.answer_prefix = answer_prefix

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[index]
        prompt_text = f"{self.prompt_prefix}{pair.source}{self.answer_prefix}"

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.source_max_length,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        answer_ids = self.tokenizer(
            pair.target,
            add_special_tokens=False,
            truncation=True,
            max_length=self.target_max_length,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            suffix = torch.cat([answer_ids, torch.tensor([eos_id])])
        else:
            suffix = answer_ids

        input_ids = torch.cat([prompt_ids, suffix])
        labels = torch.cat([torch.full_like(prompt_ids, -100), suffix])

        max_len = self.source_max_length + self.target_max_length + 1
        pad_id = self.tokenizer.pad_token_id or 0
        if len(input_ids) < max_len:
            pad_len = max_len - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id)])
            labels = torch.cat([labels, torch.full((pad_len,), -100)])

        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
        attention_mask = (input_ids != pad_id).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        losses.append(float(outputs.loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def _save_checkpoint(model, tokenizer, save_dir, device) -> None:
    # Use torch.save() which streams one tensor at a time (~128MB peak) instead of
    # safetensors which buffers the entire 1.5GB model before writing.
    # Model stays on GPU; state_dict() tensors are serialized directly from CUDA.
    torch.cuda.empty_cache()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    try:
        model.config.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        if hasattr(model, "generation_config"):
            model.generation_config.save_pretrained(save_path)
        torch.save(model.state_dict(), save_path / "pytorch_model.bin")
        print(f"checkpoint_saved={save_path}")
    except Exception as exc:
        print(f"checkpoint_save_failed={save_path} err={exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    device = resolve_device(cfg["training"]["device"])
    model_name = cfg["model"]["decoder_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensure_pad_token(tokenizer)
    torch_dtype = torch.bfloat16 if cfg["training"].get("precision") == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    if bool(cfg["training"].get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
    model = model.to(device)

    train_only_patterns = cfg["model"].get("train_only_patterns", [])
    if train_only_patterns:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name, parameter in model.named_parameters():
            if any(pat in name for pat in train_only_patterns):
                parameter.requires_grad = True

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = build_pairs_from_config(cfg=cfg, seed=seed, mask_token="[MASK]", split="train")
    val_pairs = build_pairs_from_config(cfg=cfg, seed=seed, mask_token="[MASK]", split="validation")

    prompt_prefix = cfg["model"].get("prompt_prefix", "")
    answer_prefix = cfg["model"].get("answer_prefix", " answer: ")
    source_max = int(cfg["data"]["source_max_length"])
    target_max = int(cfg["data"]["target_max_length"])

    train_dataset = CausalQADataset(
        train_pairs, tokenizer, source_max, target_max, prompt_prefix, answer_prefix
    )
    val_dataset = CausalQADataset(
        val_pairs, tokenizer, source_max, target_max, prompt_prefix, answer_prefix
    )
    train_loader = DataLoader(
        train_dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(cfg["training"]["batch_size"]), shuffle=False
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}")
    print(f"model={model_name}")
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
                    _save_checkpoint(model, tokenizer, output_dir / "best", device)
                    print(f"step={step} saved_best val_loss={best_val_loss:.4f}")

            if step % save_every == 0:
                _save_checkpoint(model, tokenizer, output_dir / f"step_{step}", device)

            if step >= max_steps:
                break

    progress.close()
    _save_checkpoint(model, tokenizer, output_dir / "final", device)
    if best_step:
        print(f"best_step={best_step} best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
