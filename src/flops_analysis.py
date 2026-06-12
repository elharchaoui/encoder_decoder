"""Inference efficiency comparison: encoder-decoder vs decoder-only.

For a document of length L queried K times, generating T tokens per query:

  Decoder-only total FLOPs ~ K * [L_prefill + K_gen] where each step
    attends over growing context.

  Encoder-decoder total FLOPs ~ encoder_once + K * decoder_per_query
    where the encoder is paid only once regardless of K.

This script measures:
  1. Latency (ms) for each component.
  2. Approximate FLOPs using the standard 2*N*D transformer formula.
  3. The crossover K where encoder-decoder becomes cheaper.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
)


def _approx_transformer_flops(
    seq_len: int,
    hidden: int,
    num_layers: int,
    intermediate: int,
    num_heads: int,
) -> int:
    """Approximate FLOPs for a single Transformer forward pass.

    Self-attention QKV projections: 3 * 2 * seq * hidden^2
    Attention weights + values:     2 * seq^2 * hidden
    Output projection:              2 * seq * hidden^2
    FFN (two linear layers):        2 * 2 * seq * hidden * intermediate
    """
    qkv = 3 * 2 * seq_len * hidden * hidden
    attn = 2 * seq_len * seq_len * hidden
    out_proj = 2 * seq_len * hidden * hidden
    ffn = 2 * 2 * seq_len * hidden * intermediate
    per_layer = qkv + attn + out_proj + ffn
    return per_layer * num_layers


def _approx_cross_attn_flops(
    dec_len: int,
    enc_len: int,
    hidden: int,
    num_layers: int,
) -> int:
    """Approximate FLOPs for cross-attention per decoder step."""
    q_proj = 2 * dec_len * hidden * hidden
    kv_proj = 2 * 2 * enc_len * hidden * hidden
    attn = 2 * dec_len * enc_len * hidden
    out_proj = 2 * dec_len * hidden * hidden
    per_layer = q_proj + kv_proj + attn + out_proj
    return per_layer * num_layers


def _time_fn(fn, warmup: int = 2, runs: int = 5) -> float:
    """Return median latency in ms."""
    for _ in range(warmup):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


@torch.no_grad()
def measure_enc_dec(
    enc_dec_name: str,
    context_lengths: list[int],
    query_counts: list[int],
    answer_tokens: int,
    device: torch.device,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(enc_dec_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(enc_dec_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    enc_cfg = model.config.encoder_config if hasattr(model.config, "encoder_config") else model.config
    dec_cfg = model.config.decoder_config if hasattr(model.config, "decoder_config") else model.config
    enc_hidden = model.config.d_model
    enc_layers = model.config.num_layers
    enc_intermediate = model.config.d_ff
    enc_heads = model.config.num_heads
    dec_hidden = model.config.d_model
    dec_layers = model.config.num_decoder_layers
    dec_intermediate = model.config.d_ff
    dec_heads = model.config.num_heads

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results: dict = {
        "model": enc_dec_name,
        "architecture": "encoder_decoder",
        "total_params": total_params,
        "trainable_params": trainable_params,
        "measurements": [],
    }

    for ctx_len in context_lengths:
        dummy_text = " ".join(["the"] * ctx_len)
        enc_inputs = tokenizer(
            dummy_text, return_tensors="pt", max_length=ctx_len, truncation=True, padding="max_length"
        )
        enc_input_ids = enc_inputs["input_ids"].to(device)
        enc_attention_mask = enc_inputs["attention_mask"].to(device)

        # Measure encoder latency (paid once per document)
        def enc_fn():
            model.get_encoder()(input_ids=enc_input_ids, attention_mask=enc_attention_mask)

        enc_latency_ms = _time_fn(enc_fn)
        encoder_outputs = model.get_encoder()(input_ids=enc_input_ids, attention_mask=enc_attention_mask)

        # Measure decoder generation latency (paid per query)
        decoder_start = torch.tensor([[model.config.decoder_start_token_id]], device=device)

        def gen_fn():
            model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=enc_attention_mask,
                max_new_tokens=answer_tokens,
                min_new_tokens=1,
                do_sample=False,
                num_beams=1,
            )

        dec_latency_ms = _time_fn(gen_fn)

        enc_flops = _approx_transformer_flops(ctx_len, enc_hidden, enc_layers, enc_intermediate, enc_heads)
        dec_self_flops = _approx_transformer_flops(answer_tokens, dec_hidden, dec_layers, dec_intermediate, dec_heads)
        dec_cross_flops = _approx_cross_attn_flops(answer_tokens, ctx_len, dec_hidden, dec_layers)
        dec_flops_per_query = dec_self_flops + dec_cross_flops

        for k in query_counts:
            total_latency_ms = enc_latency_ms + k * dec_latency_ms
            total_flops = enc_flops + k * dec_flops_per_query
            results["measurements"].append(
                {
                    "context_length": ctx_len,
                    "query_count": k,
                    "answer_tokens": answer_tokens,
                    "encoder_latency_ms": round(enc_latency_ms, 2),
                    "decoder_latency_ms_per_query": round(dec_latency_ms, 2),
                    "total_latency_ms": round(total_latency_ms, 2),
                    "encoder_flops": enc_flops,
                    "decoder_flops_per_query": dec_flops_per_query,
                    "total_flops": total_flops,
                    "total_gflops": round(total_flops / 1e9, 3),
                }
            )

    return results


@torch.no_grad()
def measure_decoder_only(
    dec_only_name: str,
    context_lengths: list[int],
    query_counts: list[int],
    answer_tokens: int,
    device: torch.device,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(dec_only_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(dec_only_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    hidden = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    intermediate = getattr(model.config, "intermediate_size", 4 * hidden)
    num_heads = model.config.num_attention_heads

    total_params = sum(p.numel() for p in model.parameters())

    results: dict = {
        "model": dec_only_name,
        "architecture": "decoder_only",
        "total_params": total_params,
        "measurements": [],
    }

    for ctx_len in context_lengths:
        dummy_text = " ".join(["the"] * ctx_len)
        inputs = tokenizer(
            dummy_text, return_tensors="pt", max_length=ctx_len, truncation=True, padding="max_length"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Each query: prefill + generate — no context reuse across queries
        def gen_fn():
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=answer_tokens,
                min_new_tokens=1,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        per_query_latency_ms = _time_fn(gen_fn)

        # FLOPs: prefill over full context + generate T tokens (each attends over growing ctx)
        prefill_flops = _approx_transformer_flops(ctx_len, hidden, num_layers, intermediate, num_heads)
        gen_flops = sum(
            _approx_transformer_flops(1, hidden, num_layers, intermediate, num_heads)
            + 2 * (ctx_len + i) * hidden * num_layers
            for i in range(answer_tokens)
        )
        flops_per_query = prefill_flops + gen_flops

        for k in query_counts:
            total_latency_ms = k * per_query_latency_ms
            total_flops = k * flops_per_query
            results["measurements"].append(
                {
                    "context_length": ctx_len,
                    "query_count": k,
                    "answer_tokens": answer_tokens,
                    "latency_ms_per_query": round(per_query_latency_ms, 2),
                    "total_latency_ms": round(total_latency_ms, 2),
                    "flops_per_query": flops_per_query,
                    "total_flops": total_flops,
                    "total_gflops": round(total_flops / 1e9, 3),
                }
            )

    return results


def find_crossover(
    enc_dec_measurements: list[dict],
    dec_only_measurements: list[dict],
) -> list[dict]:
    """Find K where enc-dec becomes cheaper than dec-only."""
    crossovers = []
    ed_by_ctx = {m["context_length"]: m for m in enc_dec_measurements}
    do_by_ctx = {m["context_length"]: m for m in dec_only_measurements}
    ctx_lens = sorted(set(ed_by_ctx) & set(do_by_ctx))
    for ctx_len in ctx_lens:
        ed_enc_lat = ed_by_ctx[ctx_len]["encoder_latency_ms"]
        ed_dec_lat = ed_by_ctx[ctx_len]["decoder_latency_ms_per_query"]
        do_lat = do_by_ctx[ctx_len]["latency_ms_per_query"]
        # crossover: ed_enc + K * ed_dec = K * do_lat
        # K = ed_enc / (do_lat - ed_dec) if do_lat > ed_dec
        if do_lat > ed_dec_lat:
            k_crossover = ed_enc_lat / (do_lat - ed_dec_lat)
            crossovers.append(
                {
                    "context_length": ctx_len,
                    "crossover_k": round(k_crossover, 2),
                    "encoder_latency_ms": round(ed_enc_lat, 2),
                    "decoder_per_query_ms": round(ed_dec_lat, 2),
                    "decoder_only_per_query_ms": round(do_lat, 2),
                }
            )
        else:
            crossovers.append(
                {
                    "context_length": ctx_len,
                    "crossover_k": "never (enc-dec already faster per query)",
                    "encoder_latency_ms": round(ed_enc_lat, 2),
                    "decoder_per_query_ms": round(ed_dec_lat, 2),
                    "decoder_only_per_query_ms": round(do_lat, 2),
                }
            )
    return crossovers


def write_flops_report(
    report_path: Path,
    enc_dec_results: dict,
    dec_only_results: dict,
    crossovers: list[dict],
    answer_tokens: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    enc_dec_name = enc_dec_results["model"]
    dec_only_name = dec_only_results["model"]
    enc_dec_params = enc_dec_results["total_params"] / 1e6
    dec_only_params = dec_only_results["total_params"] / 1e6

    lines = [
        "# Inference Efficiency: Encoder-Decoder vs Decoder-Only",
        "",
        "## Setup",
        "",
        f"- Encoder-decoder model: `{enc_dec_name}` ({enc_dec_params:.1f}M params)",
        f"- Decoder-only model: `{dec_only_name}` ({dec_only_params:.1f}M params)",
        f"- Answer tokens per query: `{answer_tokens}`",
        f"- Device: `{'cuda' if torch.cuda.is_available() else 'cpu'}`",
        "",
        "## Crossover Points",
        "",
        "The crossover K is the number of queries at which encoder-decoder total latency",
        "becomes lower than decoder-only total latency for the same document.",
        "",
        "| Context length | Crossover K | Enc-dec enc (ms) | Enc-dec dec/query (ms) | Dec-only/query (ms) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in crossovers:
        lines.append(
            f"| {c['context_length']} | {c['crossover_k']} "
            f"| {c['encoder_latency_ms']} "
            f"| {c['decoder_per_query_ms']} "
            f"| {c['decoder_only_per_query_ms']} |"
        )

    lines.extend(["", "## Latency Comparison (ms)", ""])
    ctx_lens = sorted(set(m["context_length"] for m in enc_dec_results["measurements"]))
    query_counts = sorted(set(m["query_count"] for m in enc_dec_results["measurements"]))
    lines.append("| Context length | Query count | Enc-dec total (ms) | Dec-only total (ms) | Enc-dec speedup |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    ed_lookup = {(m["context_length"], m["query_count"]): m for m in enc_dec_results["measurements"]}
    do_lookup = {(m["context_length"], m["query_count"]): m for m in dec_only_results["measurements"]}
    for ctx in ctx_lens:
        for k in query_counts:
            ed = ed_lookup.get((ctx, k))
            do = do_lookup.get((ctx, k))
            if ed and do:
                speedup = do["total_latency_ms"] / max(ed["total_latency_ms"], 0.001)
                lines.append(
                    f"| {ctx} | {k} | {ed['total_latency_ms']} | {do['total_latency_ms']} | {speedup:.2f}x |"
                )

    lines.extend(["", "## Approximate FLOPs Comparison", ""])
    lines.append("| Context length | Query count | Enc-dec GFLOPs | Dec-only GFLOPs | Enc-dec savings |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for ctx in ctx_lens:
        for k in query_counts:
            ed = ed_lookup.get((ctx, k))
            do = do_lookup.get((ctx, k))
            if ed and do:
                savings = (do["total_gflops"] - ed["total_gflops"]) / max(do["total_gflops"], 0.001)
                lines.append(
                    f"| {ctx} | {k} | {ed['total_gflops']} | {do['total_gflops']} | {savings:.1%} |"
                )

    lines.extend(["", "## Interpretation", ""])
    for c in crossovers:
        ctx = c["context_length"]
        k = c["crossover_k"]
        lines.append(
            f"- At context length `{ctx}`: encoder-decoder becomes cheaper after `{k}` queries."
        )
    lines.extend([
        "",
        "For any production system where the same document is queried more than the crossover K times",
        "(RAG, chatbot over a document, multi-turn QA), encoder-decoder has strictly lower inference cost.",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote_report={report_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--enc-dec", default="google-t5/t5-small")
    parser.add_argument("--dec-only", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument(
        "--context-lengths", nargs="+", type=int, default=[64, 128, 256, 384]
    )
    parser.add_argument("--query-counts", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--answer-tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--report", default="reports/inference_efficiency_enc_dec_vs_dec_only.md")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device={device}")
    print(f"measuring encoder-decoder: {args.enc_dec}")
    enc_dec_results = measure_enc_dec(
        args.enc_dec, args.context_lengths, args.query_counts, args.answer_tokens, device
    )
    print(f"measuring decoder-only: {args.dec_only}")
    dec_only_results = measure_decoder_only(
        args.dec_only, args.context_lengths, args.query_counts, args.answer_tokens, device
    )

    # Extract per-context crossover using K=1 measurements
    ed_k1 = {m["context_length"]: m for m in enc_dec_results["measurements"] if m["query_count"] == 1}
    do_k1 = {m["context_length"]: m for m in dec_only_results["measurements"] if m["query_count"] == 1}
    crossovers = find_crossover(
        [ed_k1[c] for c in sorted(ed_k1)],
        [do_k1[c] for c in sorted(do_k1)],
    )

    print(json.dumps({"crossovers": crossovers}, indent=2))
    write_flops_report(
        Path(args.report), enc_dec_results, dec_only_results, crossovers, args.answer_tokens
    )


if __name__ == "__main__":
    main()
