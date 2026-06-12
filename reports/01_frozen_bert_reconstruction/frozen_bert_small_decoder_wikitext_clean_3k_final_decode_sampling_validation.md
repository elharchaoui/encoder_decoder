# Experiment Evaluation Report

## Run

- Config: `configs/frozen_bert_small_decoder_wikitext_clean_3k.yaml`
- Checkpoint: `runs/frozen_bert_small_decoder_wikitext_clean_3k/final.pt`
- Data source: `wikitext`
- Encoder: `google-bert/bert-base-uncased`
- Decoder layers: `4`
- Decoder heads: `8`
- Encoder frozen: `true`
- Decoder embedding init from encoder: `True`
- Token embeddings tied: `True`

## Metrics

| Metric | Value |
| --- | ---: |
| `validation_loss` | 7.0079 |
| `perplexity` | 1105.30 |
| `generation_examples` | 64.0000 |
| `exact_match` | 0.0000 |
| `token_f1` | 0.2281 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.5024 |
| `encoder_calls_per_generation` | 1.0000 |
| `token_f1_gain_over_source_copy` | -0.2742 |
| `decode_do_sample` | 1.0000 |
| `decode_temperature` | 0.8000 |
| `decode_top_k` | 50.0000 |
| `decode_top_p` | 0.9000 |
| `decode_repetition_penalty` | 1.2000 |
| `decode_no_repeat_ngram_size` | 3.0000 |

## Samples

### Sample 1

- Source: `[MASK] jack [MASK] was [MASK] director [MASK] 's way [MASK] [MASK] scene and commented  [MASK] brief  but realistic `
- Target: `' s lead designer , jack mccornack was impressed by director lee tamahori 's way of conducting the switchblade scene and commented , " it 's brief , but realistic .`
- Prediction: `david for the ace michael, david who s ' s smart ' do ' s who suggested, but and a ball of writing, context and acting.`
- Exact match: `0.000`
- Token F1: `0.211`

### Sample 2

- Source: `[MASK] rare grade 9 player  he [MASK] one of knights [MASK] [MASK] [MASK] rushers year `
- Target: `despite being a rare grade 9 player on the senior team , he was one of the knights ' two leading rushers that year .`
- Prediction: `a addition, a an virginia to also an an batsman as a former season.`
- Exact match: `0.000`
- Token F1: `0.051`

### Sample 3

- Source: `although [MASK] toured canada from time  to [MASK] canadian [MASK] new  [MASK] held a negative [MASK] canada 's treatment [MASK] [MASK] [MASK] canadians`
- Target: `although she toured canada from time to time , and attended performances of canadian music in new york , she held a negative opinion of canada 's treatment of native musicians , saying " canadians .`
- Prediction: `although then the united in the press for a states and to first, a film as american times about by the american politicianss of norway and peoplely as society.`
- Exact match: `0.000`
- Token F1: `0.185`

### Sample 4

- Source: `22 a from [MASK] @@ [MASK] [MASK] guns hit tamura 's`
- Target: `at about 22 : 00 , a barrage from twelve 105 mm ( 4 @.@ 1 in ) guns hit tamura 's position .`
- Prediction: `on 23 1, @ 4 11 7 8 9 guns. s s gun 30 30 back ' s s t.`
- Exact match: `0.000`
- Token F1: `0.045`

### Sample 5

- Source: `it was angled at [MASK] degrees supported m ( ft ) long  [MASK] [MASK] legs [MASK] ( 72 ) above `
- Target: `it was angled at 30 degrees and supported by five 25 m ( 82 ft ) long , tapered steel legs which connected to the spikes 22 m ( 72 ft ) above the ground .`
- Prediction: `the main was the five forward ( 72 of the c in 11. @ / @ - ( long ) from m : ( ( ( 10 ( 8 cm ( hurdles of the transverse 285 ) at a centre ) (.`
- Exact match: `0.000`
- Token F1: `0.308`
