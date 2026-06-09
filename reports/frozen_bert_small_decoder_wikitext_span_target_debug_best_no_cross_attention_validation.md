# Experiment Evaluation Report

## Run

- Config: `configs/frozen_bert_small_decoder_wikitext_span_target_debug.yaml`
- Checkpoint: `runs/frozen_bert_small_decoder_wikitext_span_target_debug/best.pt`
- Data source: `wikitext`
- Data objective: `span_target`
- Encoder: `google-bert/bert-base-uncased`
- Decoder layers: `4`
- Decoder heads: `8`
- Encoder frozen: `true`
- Decoder embedding init from encoder: `True`
- Token embeddings tied: `True`
- Cross-attention enabled: `False`

## Metrics

| Metric | Value |
| --- | ---: |
| `validation_loss` | 6.7268 |
| `perplexity` | 834.50 |
| `generation_examples` | 64.0000 |
| `exact_match` | 0.0000 |
| `token_f1` | 0.0632 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.0532 |
| `source_target_token_f1` | 0.0532 |
| `prediction_source_token_f1` | 0.0556 |
| `prediction_source_copy_ratio` | 0.2227 |
| `encoder_calls_per_generation` | 1.0000 |
| `token_f1_gain_over_source_copy` | 0.0101 |
| `decode_do_sample` | 1.0000 |
| `decode_temperature` | 0.8000 |
| `decode_top_k` | 50.0000 |
| `decode_top_p` | 0.9000 |
| `decode_repetition_penalty` | 1.1000 |
| `decode_no_repeat_ngram_size` | 2.0000 |
| `decode_num_beams` | 1.0000 |

## Samples

### Sample 1

- Source: `bostaph could not [unused1] revolutions the guitar riff goes before the bass sequence .`
- Target: `tell how many`
- Prediction: `and the and`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.333`

### Sample 2

- Source: `the song was recorded in december 1938 for the library of congress [unused1] @-@ recorded in 1939 for commercial release .`
- Target: `and re`
- Prediction: `on and he`
- Exact match: `0.000`
- Token F1: `0.400`
- Prediction-source copy ratio: `0.000`

### Sample 3

- Source: `transnational route which runs north through columbus , mississippi , [unused1] and south through quitman , mississippi , to mobile , alabama , and the gulf of mexico .`
- Target: `to the us @-@ canada border`
- Prediction: `@ had`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 4

- Source: `" drymon said , " the [unused1] to mr.`
- Target: `scene where patrick is running`
- Prediction: `, for`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.500`

### Sample 5

- Source: `in october 1927 , hmas adelaide was called to the [unused1] islands protectorate as part of a punitive expedition in response to the killing of a district officer and sixteen others by kwaio natives at sinalagu on the island`
- Target: `british solomon`
- Prediction: `on,`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 6

- Source: `according to marco polo 's account , in [unused1] , the turkish and mongol horsemen " took such fright at the sight of the elephants that they would not be got to face the foe`
- Target: `the early stages of the battle`
- Prediction: `he "`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.500`

### Sample 7

- Source: `in 1469 shetland came under nominal scottish [unused1] ' lairds of norway ' kept their papa stour estates until the 17th century .`
- Target: `control , although the norse`
- Prediction: `is that`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`

### Sample 8

- Source: `the port features widescreen graphics , the optional ability to [unused1] attack mode , and the unlockable option to play as tails or knuckles the echidna .`
- Target: `spin dash , a time`
- Prediction: `were,`
- Exact match: `0.000`
- Token F1: `0.000`
- Prediction-source copy ratio: `0.000`
