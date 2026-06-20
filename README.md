# Encoder-Decoder vs Decoder-Only Context Experiments

This repository contains experiments comparing encoder-decoder and decoder-only
architectures for context-grounded generation.

The central question is:

> For tasks where a model must answer from a supplied context passage, does an
> encoder-decoder architecture provide a better quality, latency, and parameter
> tradeoff than a decoder-only model that consumes the document as a causal
> prefix?

The work focuses on extractive and multi-hop QA settings, using T5 as the
encoder-decoder family and GPT-2/Qwen2 as decoder-only baselines.

## Main Idea

Decoder-only models attend over the full prefix during generation:

```text
large input context + generated tokens -> growing KV cache -> next token
```

Encoder-decoder models separate context understanding from answer generation:

```text
large input context -> encoder hidden states -> decoder cross-attention -> answer
```

This makes the encoder output act like a reusable semantic memory. The decoder
can attend to this memory while generating, instead of repeatedly attending over
the whole input prefix.

## Headline Results

The strongest retained results are summarized in
[`implemented_experiment_plan.md`](implemented_experiment_plan.md).

Small-scale SQuAD:

| Model | Architecture | Token F1 | Exact Match |
| --- | --- | ---: | ---: |
| T5-small cross-attention-only | encoder-decoder | 0.7075 | 0.5293 |
| T5-small full fine-tune | encoder-decoder | 0.7307 | 0.5566 |
| GPT-2-small full fine-tune | decoder-only | 0.2669 | 0.1504 |
| Qwen2-0.5B full fine-tune | decoder-only | 0.5788 | 0.4004 |

Large-scale SQuAD:

| Model | Architecture | Token F1 |
| --- | --- | ---: |
| T5-large cross-attention-only | encoder-decoder | 0.8128 |
| T5-large LoRA r=8 | encoder-decoder | 0.8152 |
| T5-large full fine-tune | encoder-decoder | 0.8162 |
| GPT-2-large full fine-tune | decoder-only | 0.5041 |

Context-length scaling:

| Context tokens | T5-small XA F1 | GPT-2-small F1 |
| ---: | ---: | ---: |
| 128 | 0.5361 | 0.0872 |
| 256 | 0.7015 | 0.2373 |
| 384 | 0.7075 | 0.2669 |
| 512 | 0.7225 | 0.2884 |

Latency experiments found that T5-small generation latency stayed nearly flat
from 64 to 512 context tokens, while GPT-2-small latency grew substantially over
the same range.

## Repository Layout

```text
configs/                         Experiment configs
src/                             Training, evaluation, latency, and analysis code
reports/                         Generated experiment reports
paper/                           LaTeX paper and compiled PDF
assets/                          LinkedIn/post visuals and exported diagrams
runs/                            Local checkpoints and training outputs
implemented_experiment_plan.md   Current experiment summary and framing
```

## Setup

This project uses Python 3.10+ and `uv`.

```bash
uv sync
```

For development dependencies:

```bash
uv sync --extra dev
```

Most experiment configs assume CUDA and bfloat16. You can override device in
some evaluation and analysis commands, but training configs may need to be edited
for CPU or non-CUDA environments.

## Training

Train an encoder-decoder model:

```bash
uv run python -m src.train_seq2seq \
  --config configs/t5_small_cross_attention_only_squad_3k_seed37.yaml
```

Train a decoder-only model:

```bash
uv run python -m src.train_decoder_only \
  --config configs/gpt2_small_squad_3k_seed37.yaml
```

Checkpoints are written to the `training.output_dir` path specified in each YAML
config, usually under `runs/`.

## Evaluation

Evaluate an encoder-decoder checkpoint:

```bash
uv run python -m src.evaluate_seq2seq \
  --config configs/t5_small_cross_attention_only_squad_3k_seed37_eval512.yaml \
  --checkpoint runs/t5_small_cross_attention_only_squad_3k_seed37/best \
  --num-beams 4 \
  --report reports/my_t5_eval.md
```

Evaluate a decoder-only checkpoint:

```bash
uv run python -m src.evaluate_decoder_only \
  --config configs/gpt2_small_squad_3k_seed37_eval512.yaml \
  --checkpoint runs/gpt2_small_squad_3k_seed37/best \
  --num-beams 4 \
  --report reports/my_gpt2_eval.md
```

Add `--bertscore` to either evaluation command to compute semantic BERTScore
metrics.

## Latency And Efficiency

Benchmark T5-small vs GPT-2-small latency:

```bash
uv run python -m src.benchmark_latency \
  --t5-checkpoint runs/t5_small_cross_attention_only_squad_3k_seed37/best \
  --gpt2-checkpoint runs/gpt2_small_squad_3k_seed37/best \
  --output-dir reports
```

Run the approximate FLOPs and crossover analysis:

```bash
uv run python -m src.flops_analysis \
  --enc-dec google-t5/t5-small \
  --dec-only Qwen/Qwen2-0.5B-Instruct \
  --report reports/inference_efficiency_enc_dec_vs_dec_only.md
```

## Visual Assets

The `assets/` directory includes exported visuals for explaining the idea:

- `linkedin_encoder_decoder_schema.svg`
- `linkedin_encoder_decoder_schema.png`
- `linkedin_paper_schema_horizontal.png`
- `linkedin_paper_schema_horizontal_tight.png`

The tight horizontal version is intended for LinkedIn feed sharing.

## Paper

The working paper lives in [`paper/main.tex`](paper/main.tex), with a compiled
PDF at [`paper/main.pdf`](paper/main.pdf).

The paper is best read as an organized research write-up for the experiment
direction, not as a finalized peer-reviewed scientific paper.

## Notes

- Some historical experiment families were intentionally removed from the active
  framing, including frozen-BERT decoders, WikiText span recovery, and Qwen
  prefix-memory experiments.
- The retained comparison is deliberately narrower: context-grounded generation
  with medium-to-large input contexts.
- Reports under `reports/` are generated artifacts from specific runs and may
  use different validation sizes or decoding settings. Check each report header
  before comparing numbers directly.
