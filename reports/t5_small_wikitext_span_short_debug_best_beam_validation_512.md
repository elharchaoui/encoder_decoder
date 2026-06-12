# Seq2Seq Upper-Bound Evaluation Report

## Run

- Config: `configs/t5_small_wikitext_span_short_eval512.yaml`
- Checkpoint: `runs/t5_small_wikitext_span_short_debug/best`
- Data source: `wikitext`
- Data objective: `span_target`
- Seq2Seq model: `google-t5/t5-small`
- Input prefix: `recover missing span: `

## Metrics

| Metric | Value |
| --- | ---: |
| `validation_loss` | 2.9849 |
| `perplexity` | 19.78 |
| `generation_examples` | 512.0000 |
| `exact_match` | 0.0723 |
| `token_f1` | 0.2107 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.0294 |
| `prediction_source_token_f1` | 0.0546 |
| `prediction_source_copy_ratio` | 0.3924 |
| `unique_predictions` | 342.0000 |
| `top_prediction_ratio` | 0.0645 |
| `empty_prediction_ratio` | 0.0000 |
| `avg_prediction_length_tokens` | 1.6758 |
| `max_prediction_length_tokens` | 8.0000 |
| `target_len_1_token_f1` | 0.1940 |
| `target_len_1_examples` | 175.0000 |
| `target_len_2_3_token_f1` | 0.2194 |
| `target_len_2_3_examples` | 337.0000 |
| `token_f1_gain_over_source_copy` | 0.1814 |
| `decode_num_beams` | 4.0000 |

## Samples

### Sample 1

- Source: `other guest actors included betty <extra_id_0> " amber " daisy alcott , bernie mcinerney as old christopher penrose , carmen goodine as amy , ty jones as a doctor , and karin agstam as john scott 's sister`
- Target: `gilpin as loraine`
- Prediction: `jones as`
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
- Prediction: `, and`
- Exact match: `0.000`
- Token F1: `0.800`
- Prediction-source copy ratio: `0.500`

### Sample 5

- Source: `the fantail itself , with <extra_id_0> painted red , white and blue was installed shortly afterwards .`
- Target: `the blades`
- Prediction: `a color of`
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
