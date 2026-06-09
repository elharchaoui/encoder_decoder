# Experiment Plan: Frozen Encoder Memory + Autoregressive Decoder

## 1. Hypothesis

A pretrained encoder-only model can be run once on an input sequence and used as a fixed semantic memory for a lightweight autoregressive decoder that generates one token at a time without re-running the encoder.

The useful test is not whether encoder-decoder generation is possible. That has already been shown by BERT2BERT, BERT encoder plus Transformer decoder, BERT/GPT-2 warm-started encoder-decoder systems, and Hugging Face `EncoderDecoderModel`. The useful test is whether a modern or frozen encoder-only representation can support a small autoregressive decoder with acceptable quality, training cost, and inference efficiency.

## 2. Research Question

Can a frozen encoder-only model provide enough reusable information for a compact autoregressive decoder to generate fluent and task-correct text, while reducing trainable parameters and avoiding repeated encoder computation?

## 3. Prior Work Anchors

- Hugging Face supports encoder-decoder initialization through `EncoderDecoderModel.from_encoder_decoder_pretrained`, including BERT2BERT-style setups: https://huggingface.co/docs/transformers/en/model_doc/encoder-decoder
- Rothe, Narayan, and Severyn tested sequence generation models initialized from BERT, GPT-2, and RoBERTa checkpoints: https://aclanthology.org/2020.tacl-1.18/
- Liu and Lapata used pretrained BERT encoders for extractive and abstractive summarization, including an abstractive encoder-decoder setup: https://arxiv.org/abs/1908.08345

## 4. Main Claim To Test

The encoder does not need to be part of the generation loop. It should run once:

```text
encoder_memory = Encoder(source_tokens)
```

Then the decoder should generate autoregressively, one next token at a time:

```text
generated = [BOS]

while not done:
    next_token = Decoder(generated, encoder_memory)
    generated.append(next_token)
```

The decoder may repeatedly cross-attend to the fixed encoder memory, but the source encoder must not be called again during generation.

This is not a diffusion-style decoder. The decoder does not iteratively refine a full output sequence. It is trained and used as a standard causal next-token model conditioned on fixed encoder memory.

## 5. Minimum Viable Experiment

Use a denoising text reconstruction task because it is simple, cheap, and naturally aligned with an encoder-only model.

Input examples:

```text
Source: "the cat [MASK] on the mat"
Target: "the cat sat on the mat"

Source: "paris is capital france"
Target: "paris is the capital of france"

Source: "the model generates token one at time"
Target: "the model generates one token at a time"
```

This keeps the first experiment focused on architecture behavior rather than task complexity.

## 6. Model Variants

### Variant A: Frozen BERT + Small Decoder

- Encoder: `google-bert/bert-base-uncased`
- Decoder: randomly initialized 4-layer causal autoregressive Transformer decoder with cross-attention
- Encoder training: frozen
- Purpose: direct test of the hypothesis

### Variant B: Frozen Modern Encoder + Small Decoder

- Encoder: a stronger encoder-only checkpoint, for example ModernBERT, DeBERTa, E5, or BGE
- Decoder: same small decoder as Variant A
- Encoder training: frozen
- Purpose: test whether better encoder representations improve the frozen-memory setup

### Variant C: Decoder-Only Control

- Model: small GPT-style decoder trained on the target only, optionally with source prepended as text
- Purpose: measure whether cross-attending to encoder memory beats a simple causal baseline

## 7. Recommended First Launch

Start with Variant A only.

Reason:

- Variant A is the direct hypothesis test.
- It keeps the first launch focused on the core mechanism: one frozen encoder pass, then autoregressive decoder generation.
- Variant B and Variant C should wait until the pipeline is stable, otherwise failures will be hard to attribute.

## 8. Dataset

Use synthetic denoising data first.

Generate 50k to 200k examples from clean text by applying corruption functions:

- random token deletion
- random mask replacement
- light word shuffling within a small window
- punctuation removal
- article/preposition deletion
- span deletion

Target is always the original clean text.

Recommended source corpus for a quick local run:

- WikiText-2 for debugging
- WikiText-103 or OpenWebText subset for a larger run
- Any internal text corpus if licensing permits

Keep sequence length short for the first launch:

```text
source_max_length = 64
target_max_length = 64
```

## 9. Architecture Details

### Encoder

- Load pretrained encoder-only model.
- Run it once per source sequence.
- Pass `last_hidden_state` and source attention mask to decoder cross-attention.
- For frozen variants, set all encoder parameters to `requires_grad = False`.

### Decoder

The decoder must include:

- causal self-attention over generated target tokens
- cross-attention over fixed encoder hidden states
- feed-forward block
- output projection to tokenizer vocabulary
- KV cache during generation

The decoder must not use a diffusion, denoising-refinement, masked-token-prediction, or parallel-token objective. Its only generation objective is next-token prediction:

```text
p(y | x) = product_t p(y_t | y_<t, Encoder(x))
```

Recommended initial decoder size:

```text
layers = 4
hidden_size = encoder_hidden_size
attention_heads = 8
ffn_size = 4 * hidden_size
dropout = 0.1
```

For the first implementation, using Hugging Face `EncoderDecoderModel` is acceptable. For the custom hypothesis run, implement or configure the decoder so encoder outputs can be precomputed and passed into generation.

## 10. Training Setup

### Objective

Teacher-forced autoregressive cross-entropy:

```text
loss = -sum_t log p(target_t | target_<t, encoder(source))
```

Training uses shifted target tokens:

```text
decoder_input = [BOS], target_1, target_2, ..., target_{n-1}
labels        = target_1, target_2, ..., target_n, [EOS]
```

At each position, the decoder predicts exactly the next target token while attending to previous target tokens and the frozen encoder hidden states.

### Initial Hyperparameters

```text
optimizer = AdamW
batch_size = 32 effective
learning_rate_decoder = 3e-4
learning_rate_encoder = 0
warmup_steps = 1000
max_steps = 20k debug run, 100k full run
label_smoothing = 0.0 initially
gradient_clip_norm = 1.0
mixed_precision = bf16 if available, else fp16
```

### Early Debug Run

Before the real run, overfit 128 examples.

Pass criteria:

- training loss drops sharply
- generated samples resemble targets
- encoder forward count equals one per input sequence during generation
- no source re-tokenization or encoder call happens inside the decoding loop

## 11. Evaluation

### Quality Metrics

- exact match for synthetic denoising
- token-level F1
- BLEU or chrF for reconstruction quality
- validation negative log likelihood
- manual sample inspection every checkpoint

### Efficiency Metrics

- trainable parameter count
- total parameter count
- encoder forward calls per generated sequence
- generation tokens per second
- peak GPU memory
- latency for fixed batch sizes: 1, 8, 32

### Required Comparisons

For the first launch, compare Variant A against simple task controls:

```text
source_copy_score = metric(corrupted_source, target)
model_score = metric(FrozenBERTSmallDecoder, target)
quality_gain = model_score - source_copy_score
```

After Variant A works, add Variant C as a decoder-only control:

```text
cross_attention_gain = metric(FrozenBERTSmallDecoder) - metric(DecoderOnlyControl)
```

The hypothesis is promising only if Variant A uses the fixed encoder memory, beats source-copy and decoder-only controls, and keeps encoder calls to exactly one per generated sequence.

## 12. Instrumentation Requirement

Add an explicit generation test that counts encoder calls.

Expected behavior:

```text
encoder_calls == 1
decoder_calls == number_of_generated_tokens
```

This should be a unit or integration test, not just a manual assumption.

## 13. Implementation Milestones

### Milestone 1: Project Scaffold

Create:

```text
configs/
  frozen_bert_small_decoder.yaml
  frozen_modern_encoder_small_decoder.yaml
  decoder_only_control.yaml
src/
  data.py
  corruption.py
  models.py
  train.py
  generate.py
  evaluate.py
tests/
  test_generation_encoder_called_once.py
  test_overfit_tiny_batch.py
```

### Milestone 2: Data Pipeline

Implement:

- clean text loader
- corruption functions
- source/target tokenizer encoding
- label masking with `-100` for padding
- deterministic seed control

### Milestone 3: Frozen Encoder Variant

Implement frozen BERT encoder plus small decoder.

Required outputs:

- checkpoint
- validation metrics
- generated samples
- parameter count
- generation latency
- encoder-call-count test result

### Milestone 4: Controls And Extensions

After Variant A works, implement Variant C and then Variant B.

Required outputs:

- checkpoint
- validation metrics
- generated samples
- parameter count
- generation latency
- comparison against Variant A

### Milestone 5: Comparison Report

Write:

```text
reports/first_experiment_results.md
```

Include:

- setup
- dataset size
- model configs
- metric table
- latency table
- generated examples
- failure cases
- conclusion on whether the hypothesis survived the first test

## 14. Success Criteria

The hypothesis passes the first experiment if Variant A:

- reconstructs denoised text substantially better than the corrupted source-copy baseline
- improves over a decoder-only control once Variant C is added
- learns source-faithful reconstruction rather than only fluent continuation
- performs generation with exactly one encoder call per input
- keeps the encoder fully frozen

The hypothesis fails or needs revision if:

- the decoder ignores encoder memory
- outputs are fluent but not source-faithful
- it fails to improve over source-copy or decoder-only controls
- the small decoder needs to become so large that the efficiency advantage disappears

## 15. Launch Command Shape

The exact commands depend on the final codebase, but the intended flow should be:

```bash
python -m src.train --config configs/frozen_bert_small_decoder.yaml

python -m src.evaluate --checkpoint runs/frozen_bert_small_decoder/best

python -m pytest tests/test_generation_encoder_called_once.py
```

## 16. Next Decision

The next implementation step is to create the actual training scaffold. The fastest route is:

1. Use Hugging Face `datasets` and `transformers`.
2. Start with frozen BERT encoder plus small decoder.
3. Add a decoder-only control after the first model works.
4. Only after that, test stronger modern encoders.
