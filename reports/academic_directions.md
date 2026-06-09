# Academic Directions For The Frozen Encoder Autoregressive Decoder Experiment

Date: 2026-06-09

## Why This Search Was Needed

Our current experiment is stuck in a predictable region:

- The frozen BERT encoder helps.
- The randomly initialized autoregressive decoder learns something, but generation remains weak.
- Easier corruption improves validation loss but makes source-copy too strong.
- Stronger corruption weakens source-copy but becomes hard for the decoder.

The useful academic signal is that strong denoising generation systems usually do not train a random decoder from scratch while keeping only the encoder pretrained. They either pretrain the whole encoder-decoder, warm-start the decoder, or make the target objective narrower than full reconstruction.

## Sources Reviewed

- BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension  
  https://arxiv.org/abs/1910.13461
- MASS: Masked Sequence to Sequence Pre-training for Language Generation  
  https://arxiv.org/abs/1905.02450
- Leveraging Pre-trained Checkpoints for Sequence Generation Tasks  
  https://arxiv.org/abs/1907.12461
- Hugging Face summary of warm-starting encoder-decoder models  
  https://huggingface.co/blog/warm-starting-encoder-decoder
- PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization  
  https://arxiv.org/abs/1912.08777
- UL2: Unifying Language Learning Paradigms  
  https://arxiv.org/abs/2205.05131

## Key Insights

### 1. BART Supports The General Shape, But Not Our Initialization

BART is the closest canonical reference: corrupt text, encode the corrupted text bidirectionally, and autoregressively reconstruct the original text.

Relevant lesson:

- A left-to-right decoder is compatible with denoising.
- Span infilling is a better corruption family than arbitrary token deletion/masking.
- The decoder is pretrained, not random.

Implication for us:

- The architecture is reasonable.
- The weak part is likely decoder initialization and objective design.
- A randomly initialized decoder may need far more data than our debug runs.

### 2. MASS Suggests Predicting Missing Fragments, Not Full Reconstruction

MASS masks a contiguous fragment in the encoder input and trains the decoder to generate only the missing fragment.

Relevant lesson:

- The output target is the removed span, not the full original sequence.
- This removes most source-copy pressure.
- It directly trains the decoder to recover what is unavailable in the source.

Implication for us:

- Our full-reconstruction target encourages source-copy as a baseline and as a model behavior.
- A strong next experiment is `source = left_context + [MASK] + right_context`, `target = missing_span`.
- This gives a clean metric: can the decoder recover missing content from frozen BERT memory?

### 3. Warm-Started Encoder-Decoders Are A Better Baseline Than Random Decoders

Rothe, Narayan, and Severyn show that sequence generation can benefit from initializing encoder-decoder models from pretrained BERT, RoBERTa, and GPT-2 checkpoints. The Hugging Face implementation notes that BERT2BERT and RoBERTa2GPT2 are practical warm-start choices.

Relevant lesson:

- Initializing only the encoder leaves a lot of generation ability untrained.
- Warm-starting both encoder and decoder reduces the amount of task data needed.
- BERT-initialized decoders can work, but decoder-side causal behavior and cross-attention still need adaptation.

Implication for us:

- The strongest fair next baseline is not our random decoder.
- Compare:
  - frozen BERT encoder + random AR decoder,
  - frozen BERT encoder + pretrained GPT-2 style decoder with cross-attention,
  - BERT2BERT-style decoder initialized from BERT self-attention weights.

### 4. PEGASUS Suggests Task-Shaped Missing-Content Prediction

PEGASUS removes important sentences and trains the model to generate them from the remaining document.

Relevant lesson:

- The corruption should match the downstream behavior we care about.
- Removing important spans/sentences creates a harder generation task than random token deletion, but it avoids trivial copying.

Implication for us:

- If the long-term goal is semantic reconstruction/compression, random token corruption is too synthetic.
- We can remove salient spans or sentences and train the decoder to generate only those removed units.

### 5. UL2 Suggests Mixing Objectives Instead Of Betting On One

UL2 combines multiple denoising modes instead of committing to one corruption type.

Relevant lesson:

- Short-span, long-span, and causal objectives teach different behavior.
- Objective tags/modes can tell the model what type of recovery it is doing.

Implication for us:

- A single corruption policy may be the wrong bottleneck.
- A small mixture could train:
  - short span infilling,
  - long span infilling,
  - left-to-right continuation,
  - full reconstruction only as a minority objective.

## Recommended Pivot

Stop optimizing full-text reconstruction for now.

The next experiment should be a MASS-style missing-span generation task:

- Encoder input: corrupted text with one or more spans replaced by sentinel tokens.
- Decoder target: only the missing span text, not the full original sequence.
- Metric: target span token F1, exact match, and source-copy overlap.
- Baseline: source-copy should be near zero or clearly inappropriate, because the target is absent from the source.

This directly tests the core hypothesis:

> Can a frozen bidirectional encoder provide useful memory to an autoregressive decoder that generates missing text?

## Proposed Experiment: Span-Target Variant

### Data

Use WikiText sentence-like chunks, but construct examples as:

- Clean: `the committee approved the new financial proposal after a long debate`
- Source: `the committee approved [unused1] after a long debate`
- Target: `the new financial proposal`

Use BERT unused tokens as sentinels:

- `[unused1]`, `[unused2]`, ...

Why unused tokens:

- They are already in BERT's vocabulary.
- They avoid overloading `[MASK]`, which carries MLM-specific semantics.
- They support multiple missing spans later.

### Model

Keep the current Variant A initially:

- Frozen BERT encoder.
- Autoregressive Transformer decoder.
- Cross-attention enabled.
- BERT token embeddings reused/tied as currently implemented.

Do not change too many things at once. First change the objective.

### Baselines

Run:

1. Source-copy baseline.
2. Decoder-only control.
3. No-cross-attention ablation.
4. Current full-reconstruction model for reference.

Expected behavior:

- Source-copy should collapse because the target span is missing.
- Cross-attention should matter more.
- Decoder-only should perform poorly unless the span is highly predictable from language prior.

### Metrics

Keep:

- validation loss,
- perplexity,
- token F1,
- exact match.

Add:

- source-target lexical overlap,
- prediction-source copy ratio,
- target length bucket metrics,
- span position bucket metrics.

### Success Criteria

For a useful debug run:

- Cross-attention model beats decoder-only by at least `0.10` token F1.
- Cross-attention model beats source-copy by a large margin.
- Generated text is not mostly copied from the source.
- Validation loss and generation metrics improve together.

## Second Pivot If Span-Target Works

After proving the objective works, test better decoder initialization:

1. BERT-initialized causal decoder.
2. GPT-2 initialized decoder with added cross-attention.
3. Small pretrained seq2seq model as an upper-bound reference, such as BART/T5 small.

The purpose is not to abandon the frozen-encoder idea. The purpose is to find whether the failure is caused by:

- objective design,
- decoder initialization,
- data scale,
- or the frozen encoder bottleneck itself.

## Decision

The immediate next step should be to implement the MASS-style span-target data path and evaluation. This is more informative than more corruption tuning because it removes the source-copy trap and directly tests missing-content generation.

## Implementation Result

The MASS-style span-target path was implemented and tested with WikiText.

Result summary:

| Model / Control | Token F1 | Source-copy F1 |
| --- | ---: | ---: |
| Span-target Variant A | `0.0697` | `0.0532` |
| Span-target Variant A, no cross-attention at eval | `0.0632` | `0.0532` |
| Span-target decoder-only | `0.0686` | `0.0532` |

What this means:

- The source-copy trap is mostly removed.
- The benchmark is now cleaner and better aligned with the hypothesis.
- The current randomly initialized decoder still barely benefits from frozen encoder memory.

Updated direction:

- Next, test whether the bottleneck is decoder initialization by warm-starting the decoder or comparing against a small pretrained seq2seq upper bound.
- Also test an easier span curriculum, starting with 1-3 token spans before returning to 2-6 token spans.
