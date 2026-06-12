# Reranking Evaluation Report

## Run

- Config: `configs/frozen_bert_small_decoder_wikitext_span_short_debug.yaml`
- Checkpoint: `runs/frozen_bert_small_decoder_wikitext_span_short_debug/best.pt`

## Metrics

| Metric | Value |
| --- | ---: |
| `examples` | 128.0000 |
| `num_candidates` | 8.0000 |
| `accuracy` | 0.1562 |
| `mean_reciprocal_rank` | 0.3520 |
| `gold_loss` | 5.4346 |
| `best_loss` | 3.0004 |
| `gold_best_loss_margin` | 2.4342 |
| `prediction_token_f1` | 0.1862 |
| `random_accuracy_baseline` | 0.1250 |

## Samples

### Sample 1

- Source: `other guest actors included betty [unused1] " amber " daisy alcott , bernie mcinerney as old christopher penrose , carmen goodine as amy , ty jones as a doctor , and karin agstam as john scott 's sister`
- Gold: `gilpin as loraine`
- Prediction: `one of the`
- Gold rank: `8`
- Gold loss: `9.6043`
- Best loss: `3.7049`

### Sample 2

- Source: `when no reinforcements joined them , both companies went back to their original positions south of [unused1] daybreak .`
- Gold: `the ridge after`
- Prediction: `@-@`
- Gold rank: `5`
- Gold loss: `5.6389`
- Best loss: `3.4234`

### Sample 3

- Source: `the new routing followed mackinac trail instead [unused1] turning east to cedarville and north to sault ste .`
- Gold: `of`
- Prediction: `of`
- Gold rank: `1`
- Gold loss: `2.8933`
- Best loss: `2.8933`

### Sample 4

- Source: `styles , booker t , christian [unused1] rhino .`
- Gold: `cage , and`
- Prediction: `her`
- Gold rank: `3`
- Gold loss: `5.0486`
- Best loss: `3.7646`

### Sample 5

- Source: `the fantail itself , with [unused1] painted red , white and blue was installed shortly afterwards .`
- Gold: `the blades`
- Prediction: `number`
- Gold rank: `3`
- Gold loss: `5.8434`
- Best loss: `4.2902`

### Sample 6

- Source: `protonation of the enolate is sometimes not stereoselective , meaning that [unused1] be formed as mixtures of epimers .`
- Gold: `products can`
- Prediction: `with`
- Gold rank: `7`
- Gold loss: `6.4031`
- Best loss: `3.3670`

### Sample 7

- Source: `about 17 [unused1] 27 km ) east of lunga .`
- Gold: `mi (`
- Prediction: `to a`
- Gold rank: `2`
- Gold loss: `5.1606`
- Best loss: `3.3115`

### Sample 8

- Source: `it took the drivers 26 minutes to complete the laps , and the rain was so heavy that some drivers had to look out their side windows because they [unused1] see out their windshields .`
- Gold: `could not`
- Prediction: `the`
- Gold rank: `3`
- Gold loss: `4.5202`
- Best loss: `2.2341`
