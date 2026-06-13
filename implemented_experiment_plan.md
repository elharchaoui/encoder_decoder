# Encoder-Decoder vs Decoder-Only Context Experiments

## Objective

This repository evaluates whether encoder-decoder architectures are a better fit
than decoder-only architectures for context-grounded generation when the input
contains medium to large context.

The comparison is architectural:

- Encoder-decoder: T5-small and T5-large.
- Decoder-only: GPT-2-small, GPT-2-large, and Qwen2-0.5B.

The target tasks are extractive and multi-hop question answering, where the model
must answer from a supplied context passage rather than rely on open-ended
parametric generation.

## Current Hypothesis

For context-grounded generation, encoder-decoder models should have an advantage
because the encoder builds bidirectional representations of the full context
before answer generation. Decoder-only models process the same context as a
causal prefix, which is less aligned with reading and selecting evidence from a
document.

The expected advantage should appear in answer quality, quality per trainable
parameter, robustness as context length increases, and latency scaling as the
input context gets longer.

## Retained Experiment Families

### Small-Scale Semantic QA

Reports: `reports/05_semantic_qa_small_paradigm/`

| Model | Architecture | Task | Token F1 | Exact Match |
| --- | --- | --- | ---: | ---: |
| T5-small cross-attention-only | encoder-decoder | SQuAD | 0.7075 | 0.5293 |
| T5-small full fine-tune | encoder-decoder | SQuAD | 0.7307 | 0.5566 |
| GPT-2-small full fine-tune | decoder-only | SQuAD | 0.2669 | 0.1504 |
| Qwen2-0.5B full fine-tune | decoder-only | SQuAD | 0.5788 | 0.4004 |
| T5-small cross-attention-only | encoder-decoder | HotpotQA | 0.2086 | 0.1074 |
| GPT-2-small full fine-tune | decoder-only | HotpotQA | 0.0304 | 0.0039 |
| Qwen2-0.5B full fine-tune | decoder-only | HotpotQA | 0.1415 | 0.0664 |

### Large-Scale And Semantic Metrics

Reports: `reports/06_large_scale_and_bertscore/`

| Model | Architecture | Task | Token F1 | BERTScore F1 |
| --- | --- | --- | ---: | ---: |
| T5-large cross-attention-only | encoder-decoder | SQuAD | 0.8075-0.8128 | 0.7952 |
| T5-large LoRA r=8 | encoder-decoder | SQuAD | 0.8152 | not run |
| T5-large full fine-tune | encoder-decoder | SQuAD | 0.8162 | not run |
| GPT-2-large full fine-tune | decoder-only | SQuAD | 0.5041 | 0.5250 |
| T5-large cross-attention-only | encoder-decoder | HotpotQA | 0.3064 | 0.3439 |
| GPT-2-large full fine-tune | decoder-only | HotpotQA | 0.0850 | 0.0484 |

### Context-Length And Latency Scaling

Reports: `reports/07_long_context_efficiency/`

| Context Tokens | T5-small XA F1 | GPT-2-small F1 |
| ---: | ---: | ---: |
| 128 | 0.5361 | 0.0872 |
| 192 | 0.6660 | 0.1995 |
| 256 | 0.7015 | 0.2373 |
| 320 | 0.7081 | 0.2633 |
| 384 | 0.7075 | 0.2669 |
| 512 | 0.7225 | 0.2884 |

Latency summary:

- T5-small latency is approximately flat from 64 to 512 context tokens.
- GPT-2-small latency grows strongly over the same range.
- The current evidence supports a quality-latency tradeoff advantage for
  encoder-decoder at realistic context sizes.

## Retained Config Groups

Encoder-decoder configs:

- `configs/t5_small_*squad*.yaml`
- `configs/t5_small_*hotpotqa*.yaml`
- `configs/t5_large_*squad*.yaml`
- `configs/t5_large_*hotpotqa*.yaml`
- `configs/t5_large_lora_r8_squad_30k_seed37*.yaml`
- `configs/t5_small_xa_squad_ctx*.yaml`

Decoder-only configs:

- `configs/gpt2_small_*squad*.yaml`
- `configs/gpt2_small_*hotpotqa*.yaml`
- `configs/gpt2_large_*squad*.yaml`
- `configs/gpt2_large_*hotpotqa*.yaml`
- `configs/qwen_05b_*squad*.yaml`
- `configs/qwen_05b_*hotpotqa*.yaml`

## Removed Experiment Families

The following branches were removed from the active experiment set because they
do not directly support the current architectural comparison:

- frozen BERT plus custom randomly initialized decoder,
- WikiText denoising and span-recovery experiments,
- T5 WikiText span baselines,
- Qwen prefix-memory compressed-context experiments.

The retained Qwen work is the direct decoder-only baseline. Prefix-memory Qwen
was removed because it is a hybrid encoder/bridge/decoder architecture, not a
clean decoder-only comparator.

## Current Paper Framing

The defensible claim is narrower than universal encoder-decoder superiority:

> For context-grounded generation with medium to large input contexts,
> encoder-decoder architectures provide a better inductive bias and stronger
> quality-efficiency tradeoff than decoder-only architectures that consume the
> document as a causal prefix.

## Next Useful Checks

1. Keep Qwen2-0.5B as a modern decoder-only comparison point.
2. Align all headline evaluations to the same validation size and decoding
   protocol where possible.
3. Add confidence intervals or repeated-seed evaluations for the final tables.
4. Extend latency measurements beyond 512 context tokens if making a strong
   inference-crossover claim.
