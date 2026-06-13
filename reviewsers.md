# Reviewer Notes

Central place for reviewers to record feedback on the experiment process, evidence, and paper draft.

## Codex Review

### Summary

The research process is scientifically healthy: the original frozen-encoder plus random autoregressive decoder hypothesis was implemented, instrumented, tested against controls, and then narrowed after negative evidence. The strongest current direction is no longer the custom BERT decoder path; it is the pretrained encoder-decoder comparison, especially T5 cross-attention-only adaptation for context-grounded QA.

### Main Strengths

- The work moved from idea to executable experiments instead of staying conceptual.
- The encoder-call invariant was tested directly, which protects the original architectural claim.
- Negative results were accepted rather than hidden: the frozen BERT plus weak decoder path did not beat source-copy or decoder-only controls strongly enough.
- The pivot to T5 cross-attention-only is evidence-driven and produces a clearer contribution.
- The draft has a compelling thesis for context-grounded generation: bidirectional context encoding plus autoregressive answer decoding can be a better architecture than treating the whole context as a causal prefix.

### Main Concerns

- The current paper sometimes claims more than the evidence supports. The experiments mainly support T5 vs GPT-2 on context-grounded QA, not universal encoder-decoder superiority.
- Some headline comparisons should be protocol-aligned before submission, especially validation set size. Several T5-large SQuAD results appear to come from 256 generated examples while GPT-2-large and BERTScore evaluations use 512.
- The mechanism explanation should avoid saying decoder-only models cannot use later context at all. Generated answer tokens can attend to the full prefix; the more defensible point is that causal prefill gives weaker bidirectional passage representations.
- The inference claim should be framed as a quality-latency tradeoff unless direct wall-clock latency is lower. Existing latency evidence shows T5 latency is flatter, but GPT-2-small is still faster at 512 tokens in the measured setup.
- The strongest final claims need uncertainty estimates: multiple seeds or bootstrap confidence intervals over the evaluation set.

### Recommended Path

1. Narrow the main claim to context-grounded generation and short-answer reasoning over supplied evidence.
2. Re-evaluate every headline model on the same validation protocol, preferably 512 examples minimum.
3. Add bootstrap confidence intervals or 3-seed runs for the final QA comparisons.
4. Add at least one modern decoder-only baseline if the paper wants to argue beyond GPT-2.
5. Extend latency benchmarking beyond 512 tokens if claiming an actual inference crossover.
6. Move the original frozen-BERT/random-decoder work into an appendix or background section as a negative result that motivated the pivot.

### Suggested Thesis

Encoder-decoder models are not universally superior to decoder-only language models. However, for context-grounded generation, they offer a better inductive bias: the encoder reads the supplied evidence bidirectionally once, and the decoder generates a short answer from that fixed representation. The current evidence is strongest when presented under that narrower thesis.
