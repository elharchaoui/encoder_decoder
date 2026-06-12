from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.data import build_pairs_from_config
from src.prefix_memory import PrefixMemoryCausalLM, PrefixMemoryConfig
from src.prefix_memory_data import PrefixMemoryBatchConfig, PrefixMemoryDataset
from src.train import move_batch, resolve_device


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def model_from_config(cfg: dict) -> PrefixMemoryCausalLM:
    model_cfg = cfg["model"]
    return PrefixMemoryCausalLM(
        PrefixMemoryConfig(
            encoder_name=model_cfg["encoder_name"],
            decoder_name=model_cfg["decoder_name"],
            memory_tokens=int(model_cfg.get("memory_tokens", 64)),
            bridge_heads=int(model_cfg.get("bridge_heads", 8)),
            bridge_dropout=float(model_cfg.get("bridge_dropout", 0.0)),
            freeze_encoder=bool(model_cfg.get("freeze_encoder", True)),
            freeze_decoder=bool(model_cfg.get("freeze_decoder", True)),
            decoder_trainable_patterns=tuple(model_cfg.get("decoder_trainable_patterns", [])),
        )
    )


def save_bridge(model: PrefixMemoryCausalLM, output_dir: Path, cfg: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    decoder_adapter = {
        name: parameter.detach().cpu()
        for name, parameter in model.decoder.named_parameters()
        if parameter.requires_grad
    }
    torch.save(
        {
            "memory_queries": model.memory_queries.detach().cpu(),
            "memory_attention": model.memory_attention.state_dict(),
            "memory_norm": model.memory_norm.state_dict(),
            "memory_projection": model.memory_projection.state_dict(),
            "decoder_adapter": decoder_adapter,
        },
        output_dir / "bridge.pt",
    )
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_bridge(model: PrefixMemoryCausalLM, checkpoint_dir: str | Path, device: torch.device) -> None:
    state = torch.load(Path(checkpoint_dir) / "bridge.pt", map_location=device)
    model.memory_queries.data.copy_(state["memory_queries"].to(device))
    model.memory_attention.load_state_dict(state["memory_attention"])
    model.memory_norm.load_state_dict(state["memory_norm"])
    model.memory_projection.load_state_dict(state["memory_projection"])
    decoder_adapter = state.get("decoder_adapter", {})
    decoder_params = dict(model.decoder.named_parameters())
    for name, value in decoder_adapter.items():
        decoder_params[name].data.copy_(value.to(device))


def build_full_context_teacher(cfg: dict, decoder_tokenizer, device: torch.device):
    """Load the frozen full-context decoder-only teacher for KL distillation."""
    teacher_name = cfg["model"].get("teacher_name") or cfg["model"]["decoder_name"]
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name)
    if cfg["training"].get("precision") == "bf16":
        teacher = teacher.to(dtype=torch.bfloat16)
    teacher = teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher


@torch.no_grad()
def teacher_logits_batched(
    teacher,
    source_texts: list[str],
    target_texts: list[str],
    tokenizer,
    source_max_length: int,
    target_max_length: int,
    device: torch.device,
    source_prefix: str = "",
    answer_prefix: str = " answer: ",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single batched teacher forward pass, returning logits aligned to answer tokens.

    Returns:
        teacher_logits: (B, T_max, vocab) — logit[b, t] predicts answer_token[b, t]
        answer_mask:    (B, T_max) bool — True where answer token is real (not padding)
    """
    B = len(source_texts)

    # Tokenize prompts (source_prefix + source + answer_prefix)
    prompts = [f"{source_prefix}{src}{answer_prefix}" for src in source_texts]
    prompt_enc = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=source_max_length,
        padding=True,
        add_special_tokens=False,
    )
    prompt_ids = prompt_enc["input_ids"].to(device)            # (B, L_max)
    prompt_mask = prompt_enc["attention_mask"].to(device)
    prompt_lengths = prompt_mask.sum(dim=1)                    # (B,) actual lengths

    # Tokenize answers
    answer_enc = tokenizer(
        target_texts,
        return_tensors="pt",
        truncation=True,
        max_length=target_max_length,
        padding=True,
        add_special_tokens=False,
    )
    answer_ids = answer_enc["input_ids"].to(device)            # (B, T_max)
    answer_mask = answer_enc["attention_mask"].to(device)

    # Single forward pass on [prompt | answer]
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    full_mask = torch.cat([prompt_mask, answer_mask], dim=1)
    out = teacher(input_ids=full_ids, attention_mask=full_mask)
    logits = out.logits                                        # (B, L_max+T_max, vocab)

    # Gather answer logits: position prompt_lengths[b]-1+t predicts answer_ids[b,t]
    T_max = answer_ids.shape[1]
    t_idx = torch.arange(T_max, device=device).unsqueeze(0)   # (1, T_max)
    positions = (prompt_lengths - 1).unsqueeze(1) + t_idx     # (B, T_max)
    positions = positions.clamp(0, logits.shape[1] - 1)

    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, T_max)
    teacher_ans_logits = logits[b_idx, positions]              # (B, T_max, vocab)

    return teacher_ans_logits, answer_mask.bool()


@torch.no_grad()
def evaluate(model: PrefixMemoryCausalLM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            target_input_ids=batch["target_input_ids"],
            target_attention_mask=batch["target_attention_mask"],
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
    encoder_tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    decoder_tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["decoder_name"])
    ensure_pad_token(encoder_tokenizer)
    ensure_pad_token(decoder_tokenizer)
    model = model_from_config(cfg).to(device)
    if cfg["training"].get("precision") == "bf16":
        model = model.to(dtype=torch.bfloat16)
    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    distill_weight = float(cfg["training"].get("distill_weight", 0.0))
    teacher = None
    decoder_prefix_len = 0
    if distill_weight > 0.0:
        teacher = build_full_context_teacher(cfg, decoder_tokenizer, device)
        decoder_prefix = cfg["model"].get("decoder_prefix", "")
        decoder_prefix_len = len(
            decoder_tokenizer(decoder_prefix, add_special_tokens=False)["input_ids"]
        )
        print(f"distillation enabled: weight={distill_weight} teacher={cfg['model'].get('teacher_name', cfg['model']['decoder_name'])}")
        print(f"decoder_prefix_len={decoder_prefix_len} (answer logits start at memory_tokens+{decoder_prefix_len})")

    mask_token = cfg["data"].get("span_target", {}).get("sentinel_token", "[unused1]")
    train_pairs = build_pairs_from_config(cfg=cfg, seed=seed, mask_token=mask_token, split="train")
    val_pairs = build_pairs_from_config(cfg=cfg, seed=seed, mask_token=mask_token, split="validation")
    batch_cfg = PrefixMemoryBatchConfig(
        source_max_length=int(cfg["data"]["source_max_length"]),
        target_max_length=int(cfg["data"]["target_max_length"]),
    )
    train_dataset = PrefixMemoryDataset(
        train_pairs,
        encoder_tokenizer,
        decoder_tokenizer,
        batch_cfg,
        source_prefix=cfg["model"].get("source_prefix", ""),
        decoder_prefix=cfg["model"].get("decoder_prefix", ""),
    )
    val_dataset = PrefixMemoryDataset(
        val_pairs,
        encoder_tokenizer,
        decoder_tokenizer,
        batch_cfg,
        source_prefix=cfg["model"].get("source_prefix", ""),
        decoder_prefix=cfg["model"].get("decoder_prefix", ""),
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

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}")
    print(f"encoder_name={cfg['model']['encoder_name']}")
    print(f"decoder_name={cfg['model']['decoder_name']}")
    print(f"memory_tokens={cfg['model'].get('memory_tokens', 64)}")
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
                source_input_ids=batch["source_input_ids"],
                source_attention_mask=batch["source_attention_mask"],
                target_input_ids=batch["target_input_ids"],
                target_attention_mask=batch["target_attention_mask"],
                labels=batch["labels"],
            )
            ce_loss = outputs.loss

            if teacher is not None and distill_weight > 0.0:
                source_texts = batch.get("source_text", [])
                target_texts = batch.get("target_text", [])
                if source_texts and target_texts:
                    kl_temp = float(cfg["training"].get("distill_temperature", 2.0))
                    source_max = int(cfg["data"]["source_max_length"])
                    target_max = int(cfg["data"]["target_max_length"])
                    memory_len = int(cfg["model"].get("memory_tokens", 64))
                    # Single batched teacher forward; logits aligned to answer tokens.
                    # Teacher uses the same decoder_prefix as the student so the
                    # conditioning context is matched.
                    teacher_ans_logits, teacher_ans_mask = teacher_logits_batched(
                        teacher,
                        list(source_texts),
                        list(target_texts),
                        decoder_tokenizer,
                        source_max,
                        target_max,
                        device,
                        source_prefix=cfg["model"].get("source_prefix", ""),
                        answer_prefix=cfg["model"].get("decoder_prefix", " answer: "),
                    )
                    # Student answer logits: start AFTER memory tokens AND decoder prefix
                    # student_logits[:, memory_len + decoder_prefix_len - 1 + t, :] predicts answer token t
                    student_logits = outputs.logits
                    s_start = memory_len + decoder_prefix_len - 1
                    T_max = teacher_ans_logits.shape[1]
                    T_student = student_logits.shape[1] - s_start
                    ans_len = min(T_max, T_student)
                    if ans_len > 0:
                        s_log = student_logits[:, s_start : s_start + ans_len, :]
                        t_log = teacher_ans_logits[:, :ans_len, :]
                        # Mask out padding positions in KL loss
                        valid = teacher_ans_mask[:, :ans_len]  # (B, ans_len)
                        kl_per_pos = F.kl_div(
                            F.log_softmax(s_log / kl_temp, dim=-1),
                            F.softmax(t_log / kl_temp, dim=-1),
                            reduction="none",
                        ).sum(dim=-1)  # (B, ans_len)
                        kl_loss = (kl_per_pos * valid).sum() / valid.sum().clamp(min=1)
                        kl_loss = kl_loss * (kl_temp ** 2)
                        combined_loss = (1.0 - distill_weight) * ce_loss + distill_weight * kl_loss
                    else:
                        combined_loss = ce_loss
                else:
                    combined_loss = ce_loss
            else:
                combined_loss = ce_loss

            loss = combined_loss / accumulation_steps
            loss.backward()
            running_loss += float(ce_loss.item())

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
                    save_bridge(model, output_dir / "best", cfg)
                    encoder_tokenizer.save_pretrained(output_dir / "best" / "encoder_tokenizer")
                    decoder_tokenizer.save_pretrained(output_dir / "best" / "decoder_tokenizer")
                    print(f"step={step} saved_best val_loss={best_val_loss:.4f}")

            if step % save_every == 0:
                save_bridge(model, output_dir / f"step_{step}", cfg)

            if step >= max_steps:
                break
    progress.close()
    save_bridge(model, output_dir / "final", cfg)
    encoder_tokenizer.save_pretrained(output_dir / "final" / "encoder_tokenizer")
    decoder_tokenizer.save_pretrained(output_dir / "final" / "decoder_tokenizer")
    if best_step:
        print(f"best_step={best_step} best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
