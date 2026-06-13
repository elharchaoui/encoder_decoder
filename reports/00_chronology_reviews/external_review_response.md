# Review Response

## Scope Decision

The active paper scope is now:

> Encoder-decoder versus decoder-only architectures for context-grounded
> generation with medium to large input contexts.

This scope keeps Qwen2-0.5B as a decoder-only baseline, because it is a modern
decoder-only model and strengthens the architectural comparison beyond GPT-2.

## Accepted Cleanup

The following experiment branches are excluded from the active evidence base:

- custom frozen-BERT encoder plus randomly initialized decoder,
- WikiText reconstruction and span-recovery tasks,
- T5 WikiText span baselines,
- Qwen prefix-memory compressed-context experiments.

Reason:

- They do not directly test the retained architecture question.
- They add extra hypotheses about frozen encoders, synthetic span recovery, or
  soft-memory compression.
- Mixing them into the main narrative makes the result look less focused.

## Qwen Clarification

Retained Qwen:

- `qwen_05b_squad_3k_seed37`
- `qwen_05b_hotpotqa_3k_seed37`

These runs are direct decoder-only baselines.

Removed Qwen:

- `prefix_memory_qwen05_*`

Those runs use a hybrid encoder/bridge/decoder design. They are not appropriate
as clean decoder-only baselines.

## Strongest Current Evidence

### SQuAD

| Model | Architecture | Token F1 | Exact Match |
| --- | --- | ---: | ---: |
| T5-small cross-attention-only | encoder-decoder | 0.7075 | 0.5293 |
| T5-small full fine-tune | encoder-decoder | 0.7307 | 0.5566 |
| GPT-2-small full fine-tune | decoder-only | 0.2669 | 0.1504 |
| Qwen2-0.5B full fine-tune | decoder-only | 0.5788 | 0.4004 |
| T5-large cross-attention-only | encoder-decoder | 0.8128 | 0.6406 |
| GPT-2-large full fine-tune | decoder-only | 0.5041 | 0.3516 |

### HotpotQA

| Model | Architecture | Token F1 | Exact Match |
| --- | --- | ---: | ---: |
| T5-small cross-attention-only | encoder-decoder | 0.2086 | 0.1074 |
| T5-small full fine-tune | encoder-decoder | 0.2404 | 0.1348 |
| GPT-2-small full fine-tune | decoder-only | 0.0304 | 0.0039 |
| Qwen2-0.5B full fine-tune | decoder-only | 0.1415 | 0.0664 |
| T5-large cross-attention-only | encoder-decoder | 0.3064 | 0.1895 |
| GPT-2-large full fine-tune | decoder-only | 0.0850 | 0.0469 |

## Remaining Review Concerns

The final paper should still avoid overclaiming.

Recommended framing:

- Claim encoder-decoder superiority for context-grounded QA and similar tasks,
  not for all generation.
- Present GPT-2 and Qwen as decoder-only baselines from different model families.
- State that Qwen narrows the gap compared with GPT-2 but does not remove it.
- Separate quality, parameter efficiency, and latency claims.
- Add confidence intervals or repeated-seed evaluations before submission if
  these numbers become headline claims.

## Current Recommendation

Keep the repository focused on:

1. T5 encoder-decoder QA.
2. GPT-2 decoder-only QA.
3. Qwen2 decoder-only QA.
4. BERTScore semantic validation.
5. Context-length and latency scaling.

Everything else should stay removed or archived outside the active experiment
tree.
