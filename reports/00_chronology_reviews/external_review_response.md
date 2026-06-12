# External Review Response

Date: 2026-06-10

## Bottom Line

The review is correct.

The original hypothesis survives only in a narrower form:

> Pretrained generative stacks can be adapted efficiently by training only the conditioning bridge or cross-attention.

The stronger original version is not supported:

> A frozen encoder plus a newly initialized small decoder can generate well.

That path should stay paused unless the decoder is pretrained.

## What The Evidence Supports

### Strong Negative: Custom BERT Plus Weak Decoder

The custom frozen-BERT architecture has now failed in several ways:

- Random decoder generation is weak.
- BERT-layer warm-start improves teacher-forced loss slightly but not generation.
- Decoder-only and no-cross-attention controls are close to the cross-attention model.
- Reranking shows frozen-BERT cross-attention does not identify the correct missing span better than decoder-only.

This is not just a decoding bug. The decoder is not using the encoder memory strongly enough.

### Strong Positive: T5 Cross-Attention-Only

The best current result is T5-small with only encoder-decoder attention and decoder final norm trainable:

| Model | Trainable params | Token F1 | Exact match |
| --- | ---: | ---: | ---: |
| T5-small full fine-tune | `60.5M` | `0.2482` | `0.0938` |
| T5-small frozen encoder | `25.2M` | `0.2133` | `0.0625` |
| T5-small cross-attention-only | `6.3M` | `0.2553` | `0.0938` |

This is the cleanest evidence for efficient conditioning adaptation.

### Interesting But Not Yet Convincing: Qwen Prefix Memory

Prefix-memory Qwen is feasible:

- Qwen2-0.5B runs locally.
- Checkpoints are compact.
- Bridge-only adaptation trains.
- Final-layer adapter improves the artificial span benchmark.

But SQuAD exposes the core failure:

| Evaluation | Prefix-memory Qwen | Full-context Qwen baseline |
| --- | ---: | ---: |
| SQuAD generation token F1 | `0.0210` | `0.1390` |
| SQuAD rerank accuracy | `0.1406` | `0.6719` |

The compressed memory path is not preserving answer identity.

## Review Points Accepted

1. `64` generation examples is debug-only. Any winner needs at least `512` generated validation examples.
2. Token F1 is not enough because it can reward shallow high-frequency-token behavior.
3. Prefix-memory should be evaluated as retention under compression, not as a direct model-quality replacement.
4. Random WikiText spans are useful diagnostics but not sufficient semantic reasoning tests.

## Revised Success Criteria

### T5 Cross-Attention-Only

A serious result requires:

- at least `3` seeds,
- at least `512` generated validation examples,
- larger training sizes: `3k`, `10k`, `30k`,
- comparison against full fine-tune T5 and frozen-encoder T5.

### Qwen Prefix Memory

A credible compressed-memory result requires:

- at least `60-80%` of full-context Qwen reranking accuracy,
- much shorter decoder context than full-context Qwen,
- no prediction-collapse behavior,
- candidate reranking above random by a meaningful margin before free generation is trusted.

## Revised Next Steps

1. Promote T5 cross-attention-only to the main efficient baseline.
2. Run T5 cross-attention-only across `3` seeds with `512+` generation examples.
3. Scale T5 cross-attention-only to `10k` and `30k` training examples if seed variance is acceptable.
4. Pause custom BERT random-decoder work.
5. Continue Qwen prefix-memory only with distillation or contrastive answer-selection training.
6. Track collapse metrics as first-class metrics:
   - unique predictions,
   - top prediction ratio,
   - answer/source copy ratio,
   - length distribution,
   - entropy where practical,
   - candidate-rerank accuracy where candidates exist.

## First Follow-Up Completed

The seq2seq evaluator now tracks collapse diagnostics, and the existing T5 cross-attention-only checkpoint was re-evaluated on `512` validation generations.

| Metric | Value |
| --- | ---: |
| Token F1 | `0.2236` |
| Exact match | `0.0859` |
| Source-copy token F1 | `0.0294` |
| Unique predictions | `309 / 512` |
| Top prediction ratio | `0.0781` |
| Empty prediction ratio | `0.0020` |

Interpretation:

- The earlier `64`-example result was optimistic but directionally valid.
- T5 cross-attention-only remains the main efficient baseline.
- The next evidence needed is seed stability, not a new architecture.

## Seed Stability Completed

The `3k` T5 cross-attention-only setup was run across three seeds and evaluated on `512` validation generations.

| Seed | Token F1 | Exact match | Unique predictions | Top prediction ratio |
| ---: | ---: | ---: | ---: | ---: |
| `29` | `0.2236` | `0.0859` | `309 / 512` | `0.0781` |
| `31` | `0.2354` | `0.1074` | `299 / 512` | `0.0703` |
| `37` | `0.2380` | `0.1094` | `329 / 512` | `0.0820` |

Mean token F1 is `0.2323` with sample standard deviation `0.0077`.

Decision:

- Seed stability is good enough to scale this path.
- Next experiment should be T5 cross-attention-only at `10k` training examples.
- Qwen prefix-memory should remain paused until its training objective changes.

## 10k Scale-Up Completed

The first `10k` T5 cross-attention-only run improved substantially over the matching `3k` seed.

| Run | Token F1 | Exact match | Prediction-source copy ratio | Unique predictions | Top prediction ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3k seed 37 | `0.2380` | `0.1094` | `0.4553` | `329 / 512` | `0.0820` |
| 10k seed 37 | `0.3052` | `0.1484` | `0.3762` | `361 / 512` | `0.0664` |

Decision:

- The scale-up result supports promoting T5 cross-attention-only as the main efficient baseline.
- The next check should be a second `10k` seed before spending GPU time on `30k`.

## Second 10k Seed Completed

The second `10k` seed confirms that the scale-up gain is real enough to proceed.

| Run | Token F1 | Exact match | Unique predictions | Top prediction ratio |
| --- | ---: | ---: | ---: | ---: |
| 10k seed 37 | `0.3052` | `0.1484` | `361 / 512` | `0.0664` |
| 10k seed 31 | `0.2822` | `0.1426` | `324 / 512` | `0.0781` |

Mean token F1 is `0.2937`, compared with the `3k` mean of `0.2323`.

Decision:

- Proceed to one `30k` T5 cross-attention-only run.
- Keep Qwen prefix-memory paused until contrastive/distillation training is implemented.

## 30k Run Completed

The first `30k` T5 cross-attention-only run improves loss but only matches the `10k` generation band.

| Run | Token F1 | Exact match | Eval loss | Unique predictions | Top prediction ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10k seed 37 | `0.3052` | `0.1484` | `2.8488` | `361 / 512` | `0.0664` |
| 10k seed 31 | `0.2822` | `0.1426` | `2.7976` | `324 / 512` | `0.0781` |
| 30k seed 37 | `0.2985` | `0.1523` | `2.7115` | `379 / 512` | `0.0508` |

Decision:

- Do not claim a clean `30k > 10k` generation improvement yet.
- Next evidence should be matched 512-example evaluation for full T5/frozen-encoder T5 and a small decode sweep for the 30k checkpoint.

## Matched Baselines and Decode Sweep Completed

Full-T5 and frozen-encoder T5 were re-evaluated on the same 512-example protocol.

| Model | Token F1 | Exact match | Eval loss |
| --- | ---: | ---: | ---: |
| Full T5, 3k | `0.2107` | `0.0723` | `2.9849` |
| Frozen-encoder T5, 3k | `0.1977` | `0.0645` | `3.1352` |
| Cross-attention-only T5, 30k, beam 4 | `0.2985` | `0.1523` | `2.7115` |
| Cross-attention-only T5, 30k, beam 2 lp 0.8 | `0.3107` | `0.1563` | `2.7115` |

Decision:

- The current best efficient baseline is 30k cross-attention-only with beam 2.
- Best token-F1 decode: `num_beams=2`, `length_penalty=0.8`.
- Best exact-match decode: `num_beams=2`, `length_penalty=1.2`.
- Next high-value experiment is either a matched 30k full-T5 upper bound or moving the cross-attention-only recipe to QA.

## Matched 30k Full-T5 Upper Bound Completed

The matched full-T5 upper bound was trained at `30k` examples with seed `37` and the same `9000`-step schedule as the 30k cross-attention-only run.

| Model | Trainable params | Decode | Token F1 | Exact match | Eval loss | Unique predictions | Top prediction ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Cross-attention-only T5, 30k | `6.3M` | Beam 2, lp `0.8` | `0.3107` | `0.1563` | `2.7115` | `354 / 512` | `0.0586` |
| Cross-attention-only T5, 30k | `6.3M` | Beam 2, lp `1.2` | `0.3077` | `0.1602` | `2.7115` | `372 / 512` | `0.0547` |
| Full T5, 30k | `60.5M` | Beam 2, lp `0.8` | `0.3132` | `0.1582` | `2.5198` | `373 / 512` | `0.0586` |
| Full T5, 30k | `60.5M` | Beam 2, lp `1.2` | `0.3150` | `0.1641` | `2.5198` | `389 / 512` | `0.0547` |

Decision:

- Full T5 is the matched quality upper bound, but only narrowly.
- Cross-attention-only retains most of the full-T5 generation quality while training about one tenth of the parameters.
- The external-review recommendation was directionally correct: the custom BERT random-decoder path should remain paused, and pretrained generative adaptation should stay central.
- The next high-value experiment is no longer another WikiText scale-up; it should be a semantic QA comparison or a Qwen prefix-memory objective change with contrastive/distillation training.

## SQuAD Semantic QA Comparison Completed

The cross-attention-only T5 recipe was tested on SQuAD extractive QA to address the concern that random WikiText spans overemphasize memorized wording.

| Model | Trainable params | Token F1 | Exact match | Eval loss | Unique predictions | Top prediction ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cross-attention-only T5, SQuAD 3k | `6.3M` | `0.7075` | `0.5293` | `0.4208` | `507 / 512` | `0.0039` |
| Full T5, SQuAD 3k | `60.5M` | `0.7307` | `0.5566` | `0.4218` | `505 / 512` | `0.0039` |

Decision:

- The efficient pretrained-generative direction now has semantic-QA support, not only WikiText span-recovery support.
- Full T5 remains the quality upper bound, but cross-attention-only retains about `96.8%` of full-T5 token F1 with about one tenth of the trainable parameters.
- Collapse diagnostics are clean on both runs.
- The next best experiment is a SQuAD `10k` scale-up or decode sweep for exact-match optimization, not returning to custom BERT random decoders.
