from __future__ import annotations

import argparse

import torch
import yaml
from transformers import AutoTokenizer

from src.models import FrozenEncoderAutoregressiveDecoder, FrozenEncoderDecoderConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda, but CUDA is not available.")
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["encoder_name"])
    model = FrozenEncoderAutoregressiveDecoder(
        FrozenEncoderDecoderConfig(
            encoder_name=cfg["model"]["encoder_name"],
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    batch = tokenizer(
        args.text,
        max_length=int(cfg["data"]["source_max_length"]),
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    generated = model.generate(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        max_new_tokens=int(cfg["generation"]["max_new_tokens"]),
        do_sample=bool(args.do_sample or cfg["generation"].get("do_sample", False)),
        temperature=float(args.temperature or cfg["generation"].get("temperature", 1.0)),
        top_k=int(args.top_k if args.top_k is not None else cfg["generation"].get("top_k", 0)),
        top_p=float(args.top_p if args.top_p is not None else cfg["generation"].get("top_p", 1.0)),
        repetition_penalty=float(
            args.repetition_penalty
            if args.repetition_penalty is not None
            else cfg["generation"].get("repetition_penalty", 1.0)
        ),
        no_repeat_ngram_size=int(
            args.no_repeat_ngram_size
            if args.no_repeat_ngram_size is not None
            else cfg["generation"].get("no_repeat_ngram_size", 0)
        ),
    )
    print(tokenizer.decode(generated[0], skip_special_tokens=True))
    print(f"encoder_calls={model.encoder_call_count}")


if __name__ == "__main__":
    main()
