# Review: *Reconsidering Encoder-Decoder Architectures for Context-Grounded Generation*

**Reviewer:** R2 — "The Confound Hunter"
**Recommendation:** Weak Reject (main track) / Accept (workshop or arXiv) — conditional on reframing.
**Confidence:** 4/5
**Date:** 2026-06-14

---

## Summary

The paper argues that encoder-decoder models (T5) hold a *systematic, architectural*
advantage over decoder-only models (GPT-2) for context-grounded extractive QA, and that
this advantage is (a) large, (b) cheap to unlock via PEFT, and (c) efficiency-dominant at
long context. Experiments span two scales (60M/124M, 737M/774M) on SQuAD and HotpotQA,
with token-F1, BERTScore, and latency-vs-context measurements.

The work is real, internally consistent, honestly caveated, and unusually well-executed for
a single-GPU (RTX 3060) solo project. My concerns are about *what the experiments can
actually license you to claim*, not about execution quality.

---

## Bottom line

Submittable — but **reframe and right-size before a top-tier submission.** As written, the
central "architectural" claim is not separable from a pretraining-data/objective confound,
and that gap is the single thing that decides the paper's fate. Lead with the PEFT-efficiency
and latency story (the genuinely novel parts), not with the architecture verdict.

---

## Strengths

- **Crisp, falsifiable thesis** with a concrete practical hook (RAG / context-grounded QA).
- **The PEFT story is the strongest, most novel contribution.** "Cross-attention-only"
  adaptation + "LoRA r=8 (0.32%) matches full FT" is clean and structurally motivated.
  Build the paper around this.
- **Multi-axis evidence**: lexical F1 + BERTScore + latency-vs-context. The observation that
  GPT-2's BERTScore on HotpotQA falls *below* its token F1 (semantically unrelated, not just
  imprecise) is a sharp, concrete finding.
- **Honest Limitations section** that already names single-seed, extractive-only scope, and
  the pretraining confound.

---

## Major concerns

### M1. Architecture vs. pretraining is confounded — this is the make-or-break issue
The paper claims the advantage is *architectural*, but the design cannot separate
architecture from pretraining data + objective:
- T5 was pretrained on C4 (~750GB) with a **span-corruption objective that is essentially
  extractive QA**; GPT-2 (2019) saw ~40GB WebText with pure next-token prediction. T5 is
  advantaged on *exactly this task* by its pretraining, independent of encoder-decoder shape.
- The **LoRA result undercuts the architectural framing**: "0.32% of params matches full FT →
  the capability is already in the frozen weights" is strong evidence that *pretraining*, not
  the architecture per se, is doing the work.
- Section 5.5's three rebuttals are plausibility arguments, not controls. The only thing that
  settles it is a **same-corpus, same-objective** comparison (e.g., decoder-only pretrained on
  C4, or UL2-style objective ablations).
- **Action:** either soften every "architectural" claim to "architecture + aligned
  pretraining," or add one comparison that holds pretraining roughly constant.

### M2. Decoder-only baselines look under-tuned
GPT-2-large at 0.50 F1 on SQuAD is low; a reviewer will suspect the baseline was handicapped
by prompt format / recipe rather than by architecture. The configs suggest a single shared
recipe and an inconsistency (`max_new_tokens: 16` for GPT-2-large vs `32` for Qwen).
- **Action:** sweep the GPT-2 prompt template / LR / generation as hard as T5's, or explicitly
  document the tuning effort so "you didn't try hard enough on the baseline" is preempted.

### M3. Title/scope mismatch
Title says "context-grounded **generation**"; every task is **span-extractive QA** — the most
favorable possible setting for bidirectional encoding (the answer is a substring of the input).
Generative/abstractive RAG is asserted, not shown.
- **Action:** narrow the title to extractive QA, or add a genuinely generative task.

---

## Minor concerns

### m1. "6.3M trainable beats 124M" conflates trainable params with capability
T5-small's frozen ~60M encoder does the heavy lifting. The honest axis is *total* params
(and pretraining compute), not trainable. The "architecture dominates scale" subsection
overstates because of this.

### m2. Efficiency methodology is mixed
Quality uses beam=4; latency uses greedy. The "1.92× F1 per ms" headline combines a
beam-search quality number with a greedy-decode latency number — make decoding consistent.
Also note GPT-2 is *faster* below 512 tokens (6ms vs 50ms); state the crossover as the narrow
claim it is.

### m3. Single seed (n=1) for a paradigm-level claim
Gaps are large relative to typical seed variance, so survivable — but ≥3 seeds on the cheap
small-scale runs would materially strengthen it.

### m4. Novelty framing
"Encoder-decoder is good at extractive QA / bidirectional encoding helps" is close to received
wisdom (T5 was built for this; UL2 studied the axis). The *fresh* contributions are the
**PEFT efficiency angle** and the **latency-vs-context characterization** — lead with those.

---

## Recommended paths

- **Path A (fast, honest — workshop/arXiv):** Retitle toward *"A frozen pretrained T5 encoder
  is a strong, cheap context reader for extractive QA."* Pitch as a rigorous empirical
  efficiency study, not a paradigm verdict. Soften "architectural," fix m2, add 3 seeds at
  small scale, strengthen/document the GPT-2 baseline. Publishable largely as-is.
- **Path B (slower, main-track):** Control the M1 confound with one same-corpus / same-objective
  comparison. This justifies the strong "architectural" claim and the ambitious title.

---

## Questions for the authors

1. Was GPT-2 given the same per-model tuning budget (prompt format, LR, decoding) as T5?
2. Was Qwen2-0.5B fine-tuned or evaluated zero-shot for the Table 1 numbers (0.579 SQuAD)?
   The setup should be stated explicitly.
3. Can you isolate bidirectional encoding from the span-corruption objective at all?
4. Do the gaps survive ≥3 seeds at small scale?
