# Experiment Evaluation Report

## Run

- Config: `configs/frozen_bert_small_decoder_wikitext_middle_debug.yaml`
- Checkpoint: `runs/frozen_bert_small_decoder_wikitext_middle_debug/best.pt`
- Data source: `wikitext`
- Encoder: `google-bert/bert-base-uncased`
- Decoder layers: `4`
- Decoder heads: `8`
- Encoder frozen: `true`
- Decoder embedding init from encoder: `True`
- Token embeddings tied: `True`
- Cross-attention enabled: `True`

## Metrics

| Metric | Value |
| --- | ---: |
| `validation_loss` | 6.3145 |
| `perplexity` | 552.51 |
| `generation_examples` | 64.0000 |
| `exact_match` | 0.0000 |
| `token_f1` | 0.2072 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.7049 |
| `encoder_calls_per_generation` | 1.0000 |
| `token_f1_gain_over_source_copy` | -0.4977 |
| `decode_repetition_penalty` | 1.1500 |
| `decode_no_repeat_ngram_size` | 3.0000 |
| `decode_num_beams` | 4.0000 |
| `decode_length_penalty` | 0.8000 |
| `decode_min_new_tokens` | 8.0000 |

## Samples

### Sample 1

- Source: `previous ppv event sacrifice and tna next ppv event road both [MASK] a out of [MASK] chris sokol and bob  [MASK] `
- Target: `the previous ppv event sacrifice and tna 's next ppv event victory road both received a 7 out of 10 by chris sokol and bob kapur , respectively .`
- Prediction: `in the the, the the the series the the a the the of the the @, the and the the to the game the the yoko, the.`
- Exact match: `0.000`
- Token F1: `0.143`

### Sample 2

- Source: `[MASK] the aif [MASK] [MASK] belgium  leaving egypt in 1916`
- Target: `five infantry divisions of the aif saw action in france and belgium , leaving egypt in march 1916 .`
- Prediction: `in the was, the the the, the in the and in in in the in in..`
- Exact match: `0.000`
- Token F1: `0.229`

### Sample 3

- Source: `described as knocked " and " stale "  struggled compete against such strong opposition [MASK] and yorkshire scored three converted tries before [MASK] try [MASK] left the scores at – 1`
- Target: `described as " knocked about " and " stale " , the natives struggled to compete against such strong opposition , and yorkshire scored three converted tries before a try to ellison left the scores at 9 – 1 at`
- Prediction: `although, the the " " ", the their the the the and the the a the the to the the in the two the the five the the.`
- Exact match: `0.000`
- Token F1: `0.206`

### Sample 4

- Source: `she escorted convoys the russian far [MASK] [MASK] was guard ship at january to [MASK] 1918 `
- Target: `she escorted troop convoys to the russian far east and was guard ship at kamchatka from january to august 1918 .`
- Prediction: `she she she, the the the and the the in in the in the the.`
- Exact match: `0.000`
- Token F1: `0.167`

### Sample 5

- Source: `the war was of several near simultaneous [MASK] by the empire [MASK] yuan dynasty the`
- Target: `the war was one of several near simultaneous wars waged by the mongol empire and the yuan dynasty in the late 13th century .`
- Prediction: `the the was of the the the, the the and the the century in the the`
- Exact match: `0.000`
- Token F1: `0.450`
