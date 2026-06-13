# Reviewer Notes

Central place for reviewers to record feedback on the experiment process, evidence, and paper draft.

## Codex Review

### Summary

The research process is now focused on a clear architectural comparison: encoder-decoder models versus decoder-only models for context-grounded QA with medium to large input context. The strongest evidence is the T5 comparison against GPT-2 and Qwen2 decoder-only baselines.

### Main Strengths

- The work moved from idea to executable experiments instead of staying conceptual.
- The retained experiments compare model families rather than a single checkpoint pair.
- The inclusion of Qwen2-0.5B gives the decoder-only side a more modern baseline than GPT-2 alone.
- T5 cross-attention-only gives a parameter-efficient encoder-decoder setting, while full T5 provides an upper bound.
- The draft has a compelling thesis for context-grounded generation: bidirectional context encoding plus autoregressive answer decoding can be a better architecture than treating the whole context as a causal prefix.

### Main Concerns

- The current paper sometimes claims more than the evidence supports. The experiments support context-grounded QA, not universal encoder-decoder superiority.
- Some headline comparisons should be protocol-aligned before submission, especially validation set size. Several T5-large SQuAD results appear to come from 256 generated examples while GPT-2-large and BERTScore evaluations use 512.
- The mechanism explanation should avoid saying decoder-only models cannot use later context at all. Generated answer tokens can attend to the full prefix; the more defensible point is that causal prefill gives weaker bidirectional passage representations.
- The inference claim should be framed as a quality-latency tradeoff unless direct wall-clock latency is lower. Existing latency evidence shows T5 latency is flatter, but GPT-2-small is still faster at 512 tokens in the measured setup.
- The strongest final claims need uncertainty estimates: multiple seeds or bootstrap confidence intervals over the evaluation set.

### Recommended Path

1. Narrow the main claim to context-grounded generation and short-answer reasoning over supplied evidence.
2. Re-evaluate every headline model on the same validation protocol, preferably 512 examples minimum.
3. Add bootstrap confidence intervals or 3-seed runs for the final QA comparisons.
4. Keep Qwen2-0.5B as the modern decoder-only baseline if the paper wants to argue beyond GPT-2.
5. Extend latency benchmarking beyond 512 tokens if claiming an actual inference crossover.
6. Keep removed prototype branches out of the main narrative unless they are archived separately.

### Suggested Thesis

Encoder-decoder models are not universally superior to decoder-only language models. However, for context-grounded generation, they offer a better inductive bias: the encoder reads the supplied evidence bidirectionally once, and the decoder generates a short answer from that fixed representation. The current evidence is strongest when presented under that narrower thesis.
