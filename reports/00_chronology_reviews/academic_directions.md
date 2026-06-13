# Current Research Directions

## Active Question

Does the encoder-decoder architecture provide a better quality-efficiency tradeoff
than decoder-only architecture for context-grounded generation with medium to
large input context?

The current experiments answer this through short-answer QA:

- SQuAD for single-passage extractive QA.
- HotpotQA for multi-hop QA.
- SQuAD context-length sweeps from 128 to 512 tokens.

## Why This Direction Is Coherent

Encoder-decoder models encode the input context bidirectionally before decoding.
Decoder-only models consume the context as a causal prefix.

For context-grounded QA, the model must identify evidence inside a supplied
document. This makes bidirectional context encoding a plausible architectural
advantage.

## Retained Comparisons

Encoder-decoder:

- T5-small full fine-tune.
- T5-small cross-attention-only adaptation.
- T5-large full fine-tune.
- T5-large cross-attention-only adaptation.
- T5-large LoRA r=8.

Decoder-only:

- GPT-2-small full fine-tune.
- GPT-2-large full fine-tune.
- Qwen2-0.5B full fine-tune.

Qwen2-0.5B is retained because it is a modern decoder-only baseline and broadens
the comparison beyond GPT-2.

## Removed Directions

The following are no longer part of the active direction:

- frozen BERT plus custom decoder,
- WikiText reconstruction,
- WikiText missing-span recovery,
- T5 WikiText span baselines,
- prefix-memory Qwen.

These directions tested different hypotheses. They should not be mixed into the
main architectural comparison.

## Most Useful Next Work

1. Re-run any headline results that do not share the same validation size.
2. Add bootstrap confidence intervals for SQuAD and HotpotQA F1/EM.
3. Add at least one more modern decoder-only baseline if compute allows.
4. Extend latency measurements beyond 512 tokens if the paper claims a true
   latency crossover.
5. Keep Qwen as a direct decoder-only baseline, not as a prefix-memory variant.

## Paper Framing

The final claim should be:

> Encoder-decoder models are especially well-suited to context-grounded
> generation because they read the supplied evidence bidirectionally before
> generating a short answer. This yields stronger QA quality and a better
> quality-efficiency tradeoff than decoder-only prefix conditioning in the
> evaluated medium-context regime.

Avoid claiming:

- universal encoder-decoder superiority,
- that decoder-only models cannot use context,
- that prefix-memory Qwen is evidence for pure decoder-only behavior.
