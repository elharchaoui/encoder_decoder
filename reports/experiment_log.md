# Frozen Encoder Autoregressive Decoder: Experiment Log

## Objective

Test whether a pretrained encoder-only model can run once and act as fixed memory for a small autoregressive decoder that generates one token at a time.

The experiment intentionally avoids diffusion, masked-token generation, and full-sequence refinement. The decoder is a causal next-token model with cross-attention over frozen encoder hidden states.

## Current Architecture

- Encoder: `google-bert/bert-base-uncased`
- Encoder state: frozen, `requires_grad = False`
- Decoder: 4-layer causal `torch.nn.TransformerDecoder`
- Decoder hidden size: 768, matched to BERT
- Cross-attention: decoder attends to fixed BERT hidden states
- Output vocabulary: BERT tokenizer vocabulary
- Generation invariant: encoder is called once per input sequence

## Choices And Why

- Chose a frozen encoder first because it directly tests whether encoder-only representations can serve as reusable memory.
- Removed BERT2BERT as a main variant because the chosen hypothesis is not fine-tuning a full encoder-decoder baseline.
- Used denoising reconstruction as the first task because source-target alignment is clear and easy to debug.
- Used `uv` because the local project/package workflow is managed that way.
- Used local CUDA because PyTorch detects one CUDA device and training runs successfully on it.
- Added encoder-call instrumentation because the core claim requires proving the encoder is not re-run during generation.
- Added decoder embedding initialization from BERT token embeddings because the fully random decoder learned slowly on real WikiText text.
- Tied decoder input embeddings and LM head because it reduces trainable parameters and is standard for language models.

## Environment Findings

- `uv` is installed at `/home/med/.local/bin/uv`.
- `nvidia-smi` fails to communicate with the NVIDIA driver.
- PyTorch inside the `uv` environment reports CUDA available with one device.
- The local GPU is enough for the current 4-layer decoder experiments at sequence length 64 and batch size 8.

## Implemented Files

- `pyproject.toml`: project dependencies and pytest settings.
- `configs/frozen_bert_small_decoder.yaml`: synthetic smoke-test config.
- `configs/frozen_bert_small_decoder_wikitext_debug.yaml`: small WikiText generalization debug config.
- `configs/frozen_bert_small_decoder_wikitext_clean_debug.yaml`: cleaner WikiText chunks with stronger corruption.
- `configs/frozen_bert_small_decoder_wikitext_clean_3k.yaml`: longer clean WikiText run.
- `configs/frozen_bert_small_decoder_wikitext_overfit.yaml`: WikiText memorization check.
- `src/corruption.py`: deterministic text corruption utilities.
- `src/data.py`: synthetic and WikiText pair generation plus tokenized dataset.
- `src/models.py`: frozen BERT encoder plus autoregressive decoder model.
- `src/train.py`: training loop.
- `src/generate.py`: checkpoint-based generation.
- `src/evaluate.py`: validation loss, generation metrics, source-copy baseline, decoding overrides, and report writer.
- `tests/test_generation_encoder_called_once.py`: proves generation calls the encoder exactly once.

## Evaluation Reports

- `reports/frozen_bert_small_decoder_final_validation.md`
- `reports/frozen_bert_small_decoder_wikitext_debug_final_validation.md`
- `reports/frozen_bert_small_decoder_wikitext_clean_debug_final_validation.md`
- `reports/frozen_bert_small_decoder_wikitext_clean_debug_decode_penalty_validation.md`
- `reports/frozen_bert_small_decoder_wikitext_clean_debug_decode_sampling_validation.md`
- `reports/frozen_bert_small_decoder_wikitext_clean_3k_step_2000_decode_sampling_validation.md`
- `reports/frozen_bert_small_decoder_wikitext_clean_3k_final_decode_sampling_validation.md`
- `reports/frozen_bert_small_decoder_wikitext_overfit_final_train.md`
- `reports/frozen_bert_small_decoder_wikitext_overfit_final_validation.md`

## What Was Detected

### Synthetic Smoke Test

The synthetic setup trains quickly and confirms the implementation is wired correctly.

- Training completed on CUDA.
- Validation loss with the current tied-embedding architecture is `0.2105`.
- Validation perplexity is `1.23`.
- Generation exact match is `0.9063` over 32 validation examples.
- Generation token F1 is `0.9648` over 32 validation examples.
- Source-copy token F1 is `0.6950`, so the generated output beats the source-copy baseline by `0.2699` token F1.
- Generation produced the expected reconstruction for a known synthetic example.
- Encoder call count during generation was exactly `1`.

### WikiText Debug Run

The first WikiText run trained, but generated poor samples after 500 steps.

Finding:

- The model can optimize on real data, but 500 steps over 2k WikiText examples is too shallow for useful generation from a mostly random decoder.
- Validation loss after the tied-embedding debug run is `6.8568`.
- Validation perplexity is about `950.35`.
- Generation token F1 is `0.0927`.
- Source-copy token F1 is `0.7768`.
- The model is far below source-copy on this real-text setup.

Change made:

- Initialize decoder token embeddings from the frozen BERT token embeddings.
- Tie the decoder input embedding matrix and output LM head.

Why:

- This gives the autoregressive decoder a better lexical starting point.
- It reduces trainable parameters from about 85.1M to about 61.6M.

### WikiText Clean Debug Run

Change made:

- Added configurable corruption probabilities.
- Added span deletion and span masking.
- Added sentence-like chunking instead of raw WikiText lines.
- Added `configs/frozen_bert_small_decoder_wikitext_clean_debug.yaml`.

Why:

- The previous source-copy baseline was too strong because corruption was mild.
- Raw WikiText lines include arbitrary context breaks and markup fragments.
- A harder but cleaner denoising task is a better test of whether encoder memory helps generation.

Results:

- Validation loss is `6.7786`.
- Validation perplexity is about `878.88`.
- Generation token F1 is `0.0955`.
- Source-copy token F1 is `0.5024`.
- Encoder calls per generation remain exactly `1.0`.

Interpretation:

- Stronger corruption reduced source-copy token F1 from about `0.7768` to `0.5024`.
- Validation loss improved slightly compared with the earlier WikiText debug run, from `6.8568` to `6.7786`.
- Generation quality is still poor, so corruption and chunking helped the task shape but did not solve decoding/generalization.

### Decoding Tests

Change made:

- Added repetition penalty.
- Added no-repeat n-gram blocking.
- Added temperature.
- Added top-k sampling.
- Added top-p sampling.
- Added CLI/config wiring in `src.generate` and `src.evaluate`.

Why:

- Greedy decoding was producing repetitive and generic outputs.
- Teacher-forced loss alone was not enough to judge whether the decoder had usable token distributions.

Results on the clean WikiText checkpoint:

| Decode setup | Token F1 | Source-copy F1 | Comment |
| --- | ---: | ---: | --- |
| Greedy | `0.0955` | `0.5024` | Baseline generation |
| Greedy + repetition penalty `1.25` + no-repeat 3-gram | `0.1922` | `0.5024` | Better, still below source-copy |
| Sampling: temperature `0.8`, top-k `50`, top-p `0.9`, repetition penalty `1.2`, no-repeat 3-gram | `0.2134` | `0.5024` | Best so far, still below source-copy |

Interpretation:

- Decoding matters: F1 more than doubled from greedy to controlled sampling.
- Decoding alone is not enough: the model still trails source-copy by about `0.289` F1.
- The next useful run should train longer on the clean corruption task, because the current 750-step checkpoint is undertrained.

### Clean WikiText 3k Run

Change made:

- Added `configs/frozen_bert_small_decoder_wikitext_clean_3k.yaml`.
- Trained the clean WikiText setup for 3000 steps instead of 750.
- Evaluated `step_2000.pt` and `final.pt` with the controlled sampling setup.

Why:

- The 750-step run was likely undertrained.
- Decoding improved generation, so a longer run was needed before changing model architecture.

Training behavior:

- Validation loss improved from `7.0775` at step 500 to `6.4404` at step 1500.
- Validation loss then worsened to `6.4983` at step 2000 and `7.0079` at step 3000.
- This indicates overfitting after roughly step 1500.

Generation results:

| Checkpoint | Validation loss | Token F1 | Source-copy F1 |
| --- | ---: | ---: | ---: |
| Clean debug 750-step final | `6.7786` | `0.2134` | `0.5024` |
| Clean 3k step 2000 | `6.4983` | `0.2230` | `0.5024` |
| Clean 3k final | `7.0079` | `0.2281` | `0.5024` |

Interpretation:

- Longer training helped only slightly.
- The model still trails source-copy by about `0.274` token F1 at best.
- The current bottleneck is no longer just training duration.
- The next change should test whether the decoder is actually using encoder memory, by adding a decoder-only control and/or cross-attention ablation.

### Encoder-Memory Controls

Change made:

- Added `use_cross_attention` to the model and config path.
- Added `configs/decoder_only_wikitext_clean_1500.yaml`.
- Evaluated the trained Variant A `step_2000.pt` checkpoint with cross-attention disabled.
- Trained a decoder-only control for 1500 steps with cross-attention disabled from the beginning.

Why:

- We needed to separate "the decoder is learning a language prior" from "the decoder is using frozen BERT memory."
- If the controls matched Variant A, the encoder path would not be contributing enough to justify scaling.

Results:

| Model / Control | Validation loss | Token F1 | Source-copy F1 |
| --- | ---: | ---: | ---: |
| Variant A, clean 3k step 2000 | `6.4983` | `0.2230` | `0.5024` |
| Variant A, step 2000, no cross-attention at eval | `8.1111` | `0.1388` | `0.5024` |
| Decoder-only control, 1500 steps | `7.5355` | `0.1515` | `0.5024` |

Interpretation:

- Frozen BERT memory is useful. Removing it from the trained checkpoint drops token F1 from `0.2230` to `0.1388`.
- Training without cross-attention reaches only `0.1515` token F1 under the same decoding setup.
- Variant A is not just acting as a standalone language model.
- The encoder signal helps, but it is not yet strong enough to beat source-copy.

### WikiText Overfit Run

The 128-example WikiText overfit run succeeded.

Finding:

- Train loss dropped to about `0.0147`.
- Evaluated train loss is `0.0042`.
- Train perplexity is `1.00`.
- Train generation token F1 is `0.5411`.
- Train source-copy token F1 is `0.7867`.
- This proves the architecture and training loop can learn real text.
- Validation loss worsened during overfit, which means memorization works but generalization is not solved yet.
- Validation loss is `9.2867`.
- Validation perplexity is about `10793.05`.
- Validation generation token F1 is `0.1584`.
- Validation source-copy token F1 is `0.7678`.

Interpretation:

- The current architecture is functional.
- The immediate bottleneck is data quality, decoding quality, and generalization, not GPU availability or basic model wiring.
- Low training loss does not guarantee good greedy generation. The model can score teacher-forced targets but still decode repetitive or off-target text autoregressively.
- Source-copy is currently a strong baseline for WikiText denoising because the corruption preserves much of the target.
- Stronger corruption made source-copy less dominant, but the model still needs better decoding and/or more training to beat it.

## Known Limitations

- Greedy decoding is weak; controlled sampling is better but still below source-copy.
- Beam search has not been implemented yet.
- WikiText still contains odd fragments and markup remnants, even after sentence-like chunking.
- The decoder has no pretrained autoregressive self-attention weights.
- The current validation metric can be harsh because many targets are long and line-level.
- No medium-scale run has been completed yet.
- The evaluation loop generates one sample at a time, so it is slower than batched teacher-forced validation.
- The stronger corruption config reduces source-copy strength, but it may now be too destructive for a 750-step debug run.
- The decoder-only and no-cross-attention controls confirm encoder memory helps, but they do not solve the source-copy gap.

## Next Recommended Steps

1. Add a less destructive middle corruption config and compare source-copy F1 against model F1.
2. Add better autoregressive decoding support, especially beam search and length controls.
3. Save and evaluate the best validation checkpoint automatically instead of relying on fixed step checkpoints.
4. Consider pretrained autoregressive decoder initialization if the randomly initialized decoder remains weak.
5. Avoid larger training runs until the model beats source-copy or shows a much stronger encoder-memory gain.

## Current Decision

Do not scale to a large run yet.

Reason:

- Synthetic denoising confirms the architecture.
- WikiText overfit confirms the model can learn real examples.
- WikiText validation and generation metrics show the current data/decoding setup is not good enough for a meaningful scale-up.

The decoder-only control and cross-attention ablation are complete. They show that the encoder contributes useful information, but not enough. The next engineering step should be a cleaner middle-difficulty corruption setup plus better decoding/checkpoint selection before any larger run.
