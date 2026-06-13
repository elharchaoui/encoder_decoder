# Experiment Log

## Current Objective

The active objective is an architectural comparison between encoder-decoder and
decoder-only models for context-grounded generation under medium to large input
contexts.

Retained model families:

- T5-small and T5-large as encoder-decoder models.
- GPT-2-small, GPT-2-large, and Qwen2-0.5B as decoder-only models.

Retained tasks:

- SQuAD extractive QA.
- HotpotQA multi-hop QA.
- SQuAD context-length sweeps.
- Latency benchmarks for context scaling.

## Cleanup Decision

The repo previously included exploratory branches for custom frozen encoders,
WikiText span recovery, and Qwen prefix memory. Those branches were useful during
exploration, but they are not part of the active evidence base.

Removed from active reports/configs:

- frozen BERT plus custom autoregressive decoder,
- decoder-only WikiText controls,
- WikiText span-target and reconstruction tasks,
- T5 WikiText span baselines,
- Qwen prefix-memory compressed-context experiments.

Qwen itself remains in scope only as a direct decoder-only baseline.

## Retained Small-Scale QA Results

Reports:

- `reports/05_semantic_qa_small_paradigm/t5_small_cross_attention_only_squad_3k_seed37_best_beam_validation_512.md`
- `reports/05_semantic_qa_small_paradigm/t5_small_squad_3k_seed37_best_beam_validation_512.md`
- `reports/05_semantic_qa_small_paradigm/gpt2_small_squad_3k_seed37_best_beam4_validation_512.md`
- `reports/05_semantic_qa_small_paradigm/qwen_05b_squad_3k_seed37_best_beam4_validation_512.md`
- `reports/05_semantic_qa_small_paradigm/t5_small_cross_attention_only_hotpotqa_3k_seed37_best_beam4_validation_512.md`
- `reports/05_semantic_qa_small_paradigm/t5_small_hotpotqa_3k_seed37_best_beam4_validation_512.md`
- `reports/05_semantic_qa_small_paradigm/gpt2_small_hotpotqa_3k_seed37_best_beam4_validation_512.md`
- `reports/05_semantic_qa_small_paradigm/qwen_05b_hotpotqa_3k_seed37_best_beam4_validation_512.md`

| Model | Architecture | Task | Token F1 | Exact Match |
| --- | --- | --- | ---: | ---: |
| T5-small cross-attention-only | encoder-decoder | SQuAD | 0.7075 | 0.5293 |
| T5-small full fine-tune | encoder-decoder | SQuAD | 0.7307 | 0.5566 |
| GPT-2-small full fine-tune | decoder-only | SQuAD | 0.2669 | 0.1504 |
| Qwen2-0.5B full fine-tune | decoder-only | SQuAD | 0.5788 | 0.4004 |
| T5-small cross-attention-only | encoder-decoder | HotpotQA | 0.2086 | 0.1074 |
| T5-small full fine-tune | encoder-decoder | HotpotQA | 0.2404 | 0.1348 |
| GPT-2-small full fine-tune | decoder-only | HotpotQA | 0.0304 | 0.0039 |
| Qwen2-0.5B full fine-tune | decoder-only | HotpotQA | 0.1415 | 0.0664 |

Interpretation:

- T5-small is the strongest small-scale model on both QA tasks.
- Qwen2-0.5B is a useful modern decoder-only baseline and is stronger than
  GPT-2-small, but still trails T5-small on these context-grounded tasks.
- HotpotQA widens the architecture gap, which is consistent with the need to
  combine evidence across context passages.

## Retained Large-Scale Results

Reports: `reports/06_large_scale_and_bertscore/`

| Model | Architecture | Task | Token F1 | Exact Match |
| --- | --- | --- | ---: | ---: |
| T5-large cross-attention-only | encoder-decoder | SQuAD | 0.8128 | 0.6406 |
| T5-large LoRA r=8 | encoder-decoder | SQuAD | 0.8152 | 0.6445 |
| T5-large full fine-tune | encoder-decoder | SQuAD | 0.8162 | 0.6602 |
| GPT-2-large full fine-tune | decoder-only | SQuAD | 0.5041 | 0.3516 |
| T5-large cross-attention-only | encoder-decoder | HotpotQA | 0.3064 | 0.1895 |
| GPT-2-large full fine-tune | decoder-only | HotpotQA | 0.0850 | 0.0469 |

BERTScore checks:

| Model | Task | BERTScore F1 |
| --- | --- | ---: |
| T5-small cross-attention-only | SQuAD | 0.7028 |
| GPT-2-small | SQuAD | 0.2738 |
| T5-large cross-attention-only | SQuAD | 0.7952 |
| GPT-2-large | SQuAD | 0.5250 |
| T5-large cross-attention-only | HotpotQA | 0.3439 |
| GPT-2-large | HotpotQA | 0.0484 |

## Retained Context-Length And Latency Results

Reports: `reports/07_long_context_efficiency/`

| Context Tokens | T5-small XA F1 | GPT-2-small F1 |
| ---: | ---: | ---: |
| 128 | 0.5361 | 0.0872 |
| 192 | 0.6660 | 0.1995 |
| 256 | 0.7015 | 0.2373 |
| 320 | 0.7081 | 0.2633 |
| 384 | 0.7075 | 0.2669 |
| 512 | 0.7225 | 0.2884 |

Latency finding:

- T5-small latency changes little from 64 to 512 context tokens.
- GPT-2-small latency increases sharply over the same interval.
- This supports the claim that encoder-decoder models have a better
  quality-efficiency profile as context grows.

## Current Conclusion

The active evidence supports the following claim:

> Encoder-decoder models are better aligned than decoder-only models for
> context-grounded QA with medium to large input context, because they encode
> the supplied evidence bidirectionally before generating the answer.

Qwen2-0.5B should remain as a modern decoder-only baseline. Qwen prefix-memory
should remain out of the main narrative because it is not a pure decoder-only
architecture.
