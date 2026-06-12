# Seq2Seq Upper-Bound Evaluation Report

## Run

- Config: `configs/t5_small_cross_attention_only_wikitext_span_short_seed31_eval512.yaml`
- Checkpoint: `runs/t5_small_cross_attention_only_wikitext_span_short_seed31/best`
- Data source: `wikitext`
- Data objective: `span_target`
- Seq2Seq model: `google-t5/t5-small`
- Input prefix: `recover missing span: `

## Metrics

| Metric | Value |
| --- | ---: |
| `validation_loss` | 3.0189 |
| `perplexity` | 20.47 |
| `generation_examples` | 512.0000 |
| `exact_match` | 0.1074 |
| `token_f1` | 0.2354 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.0331 |
| `prediction_source_token_f1` | 0.0605 |
| `prediction_source_copy_ratio` | 0.4414 |
| `unique_predictions` | 299.0000 |
| `top_prediction_ratio` | 0.0703 |
| `empty_prediction_ratio` | 0.0000 |
| `avg_prediction_length_tokens` | 1.5293 |
| `max_prediction_length_tokens` | 7.0000 |
| `target_len_1_token_f1` | 0.2830 |
| `target_len_1_examples` | 174.0000 |
| `target_len_2_3_token_f1` | 0.2109 |
| `target_len_2_3_examples` | 338.0000 |
| `token_f1_gain_over_source_copy` | 0.2022 |
| `decode_num_beams` | 4.0000 |

## Samples

### Sample 1

- Source: `the freeway portion of route 29 ends at the intersection with lee avenue and it continues northwest <extra_id_0> delaware river as a four @-@ lane divided highway .`
- Target: `along the`
- Prediction: `the`
- Exact match: `0.000`
- Token F1: `0.667`
- Prediction-source copy ratio: `1.000`

### Sample 2

- Source: `in the <extra_id_0> 6th century , caracol seems to have allied with calakmul and defeated tikal , closing the early classic .`
- Target: `mid`
- Prediction: `early`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `1.000`

### Sample 3

- Source: `in later choruses , bono sings " blown <extra_id_0> " with the same melody , stretching the same note even longer .`
- Target: `by the wind`
- Prediction: `"`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `1.000`

### Sample 4

- Source: `rand said about the <extra_id_0> general that it " plays like a dream "`
- Target: `gameplay in`
- Prediction: `in`
- Exact match: `0.000`
- Token F1: `0.667`
- Prediction-source copy ratio: `0.000`

### Sample 5

- Source: `the timber used was 1 1 ⁄ 2 inches ( 38 mm ) thick and nine laminations were required at the centre , where the stock passes through <extra_id_0> of the windshaft .`
- Target: `the poll end`
- Prediction: `at the end`
- Exact match: `0.000`
- Token F1: `0.667`
- Prediction-source copy ratio: `0.667`

### Sample 6

- Source: `the average annual rainfall at tikal is 1 @,@ 945 millimetres <extra_id_0> 76 @.@ 6 in ) .`
- Target: `(`
- Prediction: `(`
- Exact match: `1.000`
- Token F1: `1.000`
- Prediction-source copy ratio: `0.000`

### Sample 7

- Source: `gauthier travelled to france , where she received private <extra_id_0> lessons from auguste @-@ jean dubulle of the paris conservatory .`
- Target: `voice`
- Prediction: `symphony`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 8

- Source: `they purchased five guns in texas , but encountered difficulty purchasing handguns in texas <extra_id_0> out @-@ of @-@ state identification and traveled to new mexico instead .`
- Target: `with`
- Prediction: `.`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `1.000`
