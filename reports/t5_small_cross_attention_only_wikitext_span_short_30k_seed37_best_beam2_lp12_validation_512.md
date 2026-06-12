# Seq2Seq Upper-Bound Evaluation Report

## Run

- Config: `configs/t5_small_cross_attention_only_wikitext_span_short_30k_seed37_eval512.yaml`
- Checkpoint: `runs/t5_small_cross_attention_only_wikitext_span_short_30k_seed37/best`
- Data source: `wikitext`
- Data objective: `span_target`
- Seq2Seq model: `google-t5/t5-small`
- Input prefix: `recover missing span: `

## Metrics

| Metric | Value |
| --- | ---: |
| `validation_loss` | 2.7115 |
| `perplexity` | 15.05 |
| `generation_examples` | 512.0000 |
| `exact_match` | 0.1602 |
| `token_f1` | 0.3077 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.0293 |
| `prediction_source_token_f1` | 0.0573 |
| `prediction_source_copy_ratio` | 0.3727 |
| `unique_predictions` | 372.0000 |
| `top_prediction_ratio` | 0.0547 |
| `empty_prediction_ratio` | 0.0020 |
| `avg_prediction_length_tokens` | 1.7793 |
| `max_prediction_length_tokens` | 5.0000 |
| `target_len_1_token_f1` | 0.3843 |
| `target_len_1_examples` | 180.0000 |
| `target_len_2_3_token_f1` | 0.2662 |
| `target_len_2_3_examples` | 332.0000 |
| `token_f1_gain_over_source_copy` | 0.2784 |
| `decode_num_beams` | 2.0000 |
| `decode_length_penalty` | 1.2000 |

## Samples

### Sample 1

- Source: `after their first draw of the tour ( against northumberland county <extra_id_0> they defeated stockton @-@ on @-@ tees and tynemouth .`
- Target: `)`
- Prediction: `)`
- Exact match: `1.000`
- Token F1: `1.000`
- Prediction-source copy ratio: `0.000`

### Sample 2

- Source: `she pours into the pie the right proportion of liquor , and <extra_id_0> her guests .`
- Target: `goes back to`
- Prediction: `eats it with`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 3

- Source: `almost every soldier able to walk had to help <extra_id_0> wounded .`
- Target: `carry the`
- Prediction: `be`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 4

- Source: `" back off boogaloo " also appeared on his 2007 compilation photograph : the very best of ringo starr , <extra_id_0> collector 's edition of which included his 1972 video for the song .`
- Target: `the`
- Prediction: `a`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 5

- Source: `radcliffe 's performance received positive reviews as critics <extra_id_0> impressed by the nuance and depth of his against @-@ type role .`
- Target: `were`
- Prediction: `were`
- Exact match: `1.000`
- Token F1: `1.000`
- Prediction-source copy ratio: `0.000`

### Sample 6

- Source: `brundidge ranked <extra_id_0> ranked 94th .`
- Target: `98th and burke`
- Prediction: `93rd, and`
- Exact match: `0.000`
- Token F1: `0.400`
- Prediction-source copy ratio: `0.000`

### Sample 7

- Source: `these included the imperial ming record of loyalty ( huang ming biaozhong ji , 皇明表忠紀 ) , a biography of loyal ming <extra_id_0> and the edicts of the imperial ming ( huang ming zhaozhi , 皇明詔制 ) , a`
- Target: `officials ,`
- Prediction: `,`
- Exact match: `0.000`
- Token F1: `0.667`
- Prediction-source copy ratio: `1.000`

### Sample 8

- Source: `the level design is flatter , <extra_id_0> no vertical loops , and sonic cannot re @-@ collect his rings after being hit .`
- Target: `with`
- Prediction: `with`
- Exact match: `1.000`
- Token F1: `1.000`
- Prediction-source copy ratio: `0.000`
