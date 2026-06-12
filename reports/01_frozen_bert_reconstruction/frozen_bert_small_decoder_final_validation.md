# Experiment Evaluation Report

## Run

- Config: `configs/frozen_bert_small_decoder.yaml`
- Checkpoint: `runs/frozen_bert_small_decoder/final.pt`
- Data source: `synthetic`
- Encoder: `google-bert/bert-base-uncased`
- Decoder layers: `4`
- Decoder heads: `8`
- Encoder frozen: `true`
- Decoder embedding init from encoder: `True`
- Token embeddings tied: `True`

## Metrics

| Metric | Value |
| --- | ---: |
| `validation_loss` | 0.2105 |
| `perplexity` | 1.23 |
| `generation_examples` | 32.0000 |
| `exact_match` | 0.9062 |
| `token_f1` | 0.9648 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.6950 |
| `encoder_calls_per_generation` | 1.0000 |
| `token_f1_gain_over_source_copy` | 0.2699 |

## Samples

### Sample 1

- Source: `the experiment [MASK] generated [MASK] against [MASK] clean target`
- Target: `the experiment compares generated text against the clean target.`
- Prediction: `the experiment compares generated text against the clean target.`
- Exact match: `1.000`
- Token F1: `1.000`

### Sample 2

- Source: `decoder predicts the next [MASK]`
- Target: `a small autoregressive decoder predicts the next token.`
- Prediction: `the deco lets the decoder inspect tokens and`
- Exact match: `0.000`
- Token F1: `0.250`

### Sample 3

- Source: `frozen can encoder provide reusable semantic memory`
- Target: `a frozen encoder can provide reusable semantic memory.`
- Prediction: `a frozen encoder can provide reusable semantic memory.`
- Exact match: `1.000`
- Token F1: `1.000`

### Sample 4

- Source: `the source encoder should exactly run once generation`
- Target: `the source encoder should run exactly once during generation.`
- Prediction: `the source encoder should run exactly once during generation.`
- Exact match: `1.000`
- Token F1: `1.000`

### Sample 5

- Source: `[MASK] attention decoder lets the inspect the encoded source`
- Target: `cross attention lets the decoder inspect the encoded source.`
- Prediction: `cross attention lets the decoder inspect the encoded source.`
- Exact match: `1.000`
- Token F1: `1.000`
