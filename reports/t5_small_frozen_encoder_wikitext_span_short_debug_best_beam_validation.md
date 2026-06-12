# Seq2Seq Upper-Bound Evaluation Report

## Run

- Config: `configs/t5_small_frozen_encoder_wikitext_span_short_debug.yaml`
- Checkpoint: `runs/t5_small_frozen_encoder_wikitext_span_short_debug/best`
- Data source: `wikitext`
- Data objective: `span_target`
- Seq2Seq model: `google-t5/t5-small`
- Input prefix: `recover missing span: `

## Metrics

| Metric | Value |
| --- | ---: |
| `validation_loss` | 3.0935 |
| `perplexity` | 22.05 |
| `generation_examples` | 64.0000 |
| `exact_match` | 0.0625 |
| `token_f1` | 0.2133 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.0415 |
| `prediction_source_token_f1` | 0.0731 |
| `prediction_source_copy_ratio` | 0.4583 |
| `target_len_1_token_f1` | 0.2200 |
| `target_len_1_examples` | 20.0000 |
| `target_len_2_3_token_f1` | 0.2103 |
| `target_len_2_3_examples` | 44.0000 |
| `token_f1_gain_over_source_copy` | 0.1718 |
| `decode_num_beams` | 4.0000 |
| `decode_length_penalty` | 1.0000 |
| `decode_min_new_tokens` | 1.0000 |

## Samples

### Sample 1

- Source: `other guest actors included betty <extra_id_0> " amber " daisy alcott , bernie mcinerney as old christopher penrose , carmen goodine as amy , ty jones as a doctor , and karin agstam as john scott 's sister`
- Target: `gilpin as loraine`
- Prediction: `mcinerney as`
- Exact match: `0.000`
- Token F1: `0.400`
- Prediction-source copy ratio: `1.000`

### Sample 2

- Source: `when no reinforcements joined them , both companies went back to their original positions south of <extra_id_0> daybreak .`
- Target: `the ridge after`
- Prediction: `after a`
- Exact match: `0.000`
- Token F1: `0.400`
- Prediction-source copy ratio: `0.000`

### Sample 3

- Source: `the new routing followed mackinac trail instead <extra_id_0> turning east to cedarville and north to sault ste .`
- Target: `of`
- Prediction: `,`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 4

- Source: `styles , booker t , christian <extra_id_0> rhino .`
- Target: `cage , and`
- Prediction: `,`
- Exact match: `0.000`
- Token F1: `0.500`
- Prediction-source copy ratio: `1.000`

### Sample 5

- Source: `the fantail itself , with <extra_id_0> painted red , white and blue was installed shortly afterwards .`
- Target: `the blades`
- Prediction: `a color`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 6

- Source: `protonation of the enolate is sometimes not stereoselective , meaning that <extra_id_0> be formed as mixtures of epimers .`
- Target: `products can`
- Prediction: `can`
- Exact match: `0.000`
- Token F1: `0.667`
- Prediction-source copy ratio: `0.000`

### Sample 7

- Source: `about 17 <extra_id_0> 27 km ) east of lunga .`
- Target: `mi (`
- Prediction: `(`
- Exact match: `0.000`
- Token F1: `0.667`
- Prediction-source copy ratio: `0.000`

### Sample 8

- Source: `it took the drivers 26 minutes to complete the laps , and the rain was so heavy that some drivers had to look out their side windows because they <extra_id_0> see out their windshields .`
- Target: `could not`
- Prediction: `had to`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `1.000`
