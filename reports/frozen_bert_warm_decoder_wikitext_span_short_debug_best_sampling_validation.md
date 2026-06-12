# Experiment Evaluation Report

## Run

- Config: `configs/frozen_bert_warm_decoder_wikitext_span_short_debug.yaml`
- Checkpoint: `runs/frozen_bert_warm_decoder_wikitext_span_short_debug/best.pt`
- Data source: `wikitext`
- Data objective: `span_target`
- Encoder: `google-bert/bert-base-uncased`
- Decoder layers: `4`
- Decoder heads: `8`
- Encoder frozen: `true`
- Decoder embedding init from encoder: `True`
- Decoder layer init from encoder: `True`
- Token embeddings tied: `True`
- Cross-attention enabled: `True`

## Metrics

| Metric | Value |
| --- | ---: |
| `validation_loss` | 5.5354 |
| `perplexity` | 253.52 |
| `generation_examples` | 64.0000 |
| `exact_match` | 0.0000 |
| `token_f1` | 0.0760 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.0415 |
| `source_target_token_f1` | 0.0415 |
| `prediction_source_token_f1` | 0.0448 |
| `prediction_source_copy_ratio` | 0.4062 |
| `encoder_calls_per_generation` | 1.0000 |
| `target_len_1_token_f1` | 0.0667 |
| `target_len_1_examples` | 20.0000 |
| `target_len_2_3_token_f1` | 0.0803 |
| `target_len_2_3_examples` | 44.0000 |
| `token_f1_gain_over_source_copy` | 0.0345 |
| `decode_do_sample` | 1.0000 |
| `decode_temperature` | 0.8000 |
| `decode_top_k` | 50.0000 |
| `decode_top_p` | 0.9000 |
| `decode_repetition_penalty` | 1.1000 |
| `decode_no_repeat_ngram_size` | 2.0000 |
| `decode_num_beams` | 1.0000 |

## Samples

### Sample 1

- Source: `other guest actors included betty [unused1] " amber " daisy alcott , bernie mcinerney as old christopher penrose , carmen goodine as amy , ty jones as a doctor , and karin agstam as john scott 's sister`
- Target: `gilpin as loraine`
- Prediction: `"`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `1.000`

### Sample 2

- Source: `when no reinforcements joined them , both companies went back to their original positions south of [unused1] daybreak .`
- Target: `the ridge after`
- Prediction: `the of`
- Exact match: `0.000`
- Token F1: `0.400`
- Prediction-source copy ratio: `0.500`

### Sample 3

- Source: `the new routing followed mackinac trail instead [unused1] turning east to cedarville and north to sault ste .`
- Target: `of`
- Prediction: `,`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 4

- Source: `styles , booker t , christian [unused1] rhino .`
- Target: `cage , and`
- Prediction: `the`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 5

- Source: `the fantail itself , with [unused1] painted red , white and blue was installed shortly afterwards .`
- Target: `the blades`
- Prediction: `'`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 6

- Source: `protonation of the enolate is sometimes not stereoselective , meaning that [unused1] be formed as mixtures of epimers .`
- Target: `products can`
- Prediction: `of`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `1.000`

### Sample 7

- Source: `about 17 [unused1] 27 km ) east of lunga .`
- Target: `mi (`
- Prediction: `the was`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 8

- Source: `it took the drivers 26 minutes to complete the laps , and the rain was so heavy that some drivers had to look out their side windows because they [unused1] see out their windshields .`
- Target: `could not`
- Prediction: `"`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`
