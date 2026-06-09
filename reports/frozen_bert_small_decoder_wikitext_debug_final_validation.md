# Experiment Evaluation Report

## Run

- Config: `configs/frozen_bert_small_decoder_wikitext_debug.yaml`
- Checkpoint: `runs/frozen_bert_small_decoder_wikitext_debug/final.pt`
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
| `validation_loss` | 6.8568 |
| `perplexity` | 950.35 |
| `generation_examples` | 64.0000 |
| `exact_match` | 0.0000 |
| `token_f1` | 0.0927 |
| `source_copy_exact_match` | 0.0000 |
| `source_copy_token_f1` | 0.7768 |
| `encoder_calls_per_generation` | 1.0000 |
| `token_f1_gain_over_source_copy` | -0.6841 |

## Samples

### Sample 1

- Source: `" rowin foula doon [MASK] refers to the fishermens ' [MASK] rowing open fishing boat out to sea until the high cliffs of foula were no longer [MASK] entailed the boat being some 96 kilometres ( 60 mi ) west of papa stour ' tide @-@ lumps ' increased swells are unusual size due the combined action of wind against tide  resonant image [MASK] piece is of [MASK] being led back home papa by the scent [MASK] [MASK] across water [MASK] this is an example of vagaland 's ability create [MASK] vivid sensual [MASK] of a situation an extra layer of meaning is added knowledge [MASK] da horn papa a storm around the [MASK] of this [MASK] 's composition  so that it is tribute not [MASK] lost way life  but a noted [MASK] feature `
- Target: `" rowin foula doon ! " refers to the fishermens ' practice of rowing their open fishing boat out to sea until the high cliffs of foula were no longer visible . this entailed the boat being some 96 kilometres ( 60 mi ) west of papa stour . the ' tide @-@ lumps ' are increased swells of unusual size due to the combined action of wind against tide . the resonant final image of the piece is of the fishermen being led back home to papa by the ' scent o flooers ' across the water . this is an example of vagaland 's ability to create a vivid sensual impression of a situation . an extra layer of meaning is added by the knowledge that da horn o papa collapsed in a storm around the time of this poem 's composition , so that it is a tribute not just to a lost way of life , but a noted geographical feature .`
- Prediction: `the the, the the the the the the the the @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @`
- Exact match: `0.000`
- Token F1: `0.079`

### Sample 2

- Source: `gauthier was a controversial musician in her time [MASK] her choice music for performance was often condemned  and often [MASK]  the of jazz music classically trained singer  [MASK] with the performances taking place in [MASK] halls lead some cheer her for otherwise overlooked music  and others to condemn her [MASK] lowbrow music into [MASK] highbrow venue `
- Target: `gauthier was a controversial musician in her time . her choice of music for performance was often condemned , and often praised . the appropriateness of jazz music for a classically trained singer , combined with the performances taking place in concert halls lead some critics to cheer her for promoting otherwise overlooked music , and others to condemn her for taking lowbrow music into a highbrow venue .`
- Prediction: `the the,,,,, the,, the the the the the,,,, the the the the the,,, the the,, the,,, the the the, the the,,, the the, the the, the the and the the,, the., the the`
- Exact match: `0.000`
- Token F1: `0.059`

### Sample 3

- Source: `[MASK] 1987 profile on [MASK]  the denver post reported that [MASK] was drafted  went to paratrooper school  then volunteered vietnam  where he 10 @-@ month tour as long range reconnaissance patrol [MASK] lrrp )  one of a six @-@ man team sent out to track down enemy [MASK] the post article also reported that [MASK] was politically radicalized as a result of his experiences [MASK] vietnam  [MASK] [MASK] the post he spent some time at [MASK] chicago the the students for democratic [MASK] ( [MASK] [MASK] 1960s  and briefly taught members of [MASK] underground how to build bombs and fire weapons [MASK]`
- Target: `in a 1987 profile on churchill , the denver post reported that he was drafted , went to paratrooper school , then volunteered for vietnam , where he served a 10 @-@ month tour as long range reconnaissance patrol ( lrrp ) , one of a six @-@ man team sent out to track down the enemy . the post article also reported that churchill was politically radicalized as a result of his experiences in vietnam . churchill told the post that he had spent some time at the chicago office of the students for a democratic society ( sds ) in the late 1960s , and briefly taught members of the weather underground how to build bombs and fire weapons .`
- Prediction: `the the,, the the the the the the the the the the the the the, the the the the the the the the the the @ @ @ @ @ @ @ the the the the the the @ @ @ @ @ @ the the the the @ @ @ @ @ @ @ @ @ the the`
- Exact match: `0.000`
- Token F1: `0.088`

### Sample 4

- Source: `stela 1 dates to the 5th and depicts the king siyaj [MASK] k 'awiil ii in a standing `
- Target: `stela 1 dates to the 5th century and depicts the king siyaj chan k 'awiil ii in a standing position .`
- Prediction: `the the of the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the. the the the`
- Exact match: `0.000`
- Token F1: `0.049`

### Sample 5

- Source: `[MASK] writing used their [MASK] childhood experiences [MASK] inspirations come up much of story lines for episode  the idea for " sailor mouth " was inspired creative director derek [MASK] 's experience " [ [MASK] ] i in trouble for saying f @-@ word in front my mother  " drymon said  " where is running to [MASK] to tattle  spongebob chasing him  is pretty much [MASK] it happened real  [MASK] end episode  where mr krabs uses more profanity than [MASK] and patrick  was also inspired [MASK] by fact that my [ drymon 's ] mother has a sailor mouth herself  "`
- Target: `the writing staff used their individual childhood experiences as inspirations to come up with much of the story lines for this episode . the idea for " sailor mouth " was inspired by creative director derek drymon 's experience " [ when ] i got in trouble for saying the f @-@ word in front of my mother . " drymon said , " the scene where patrick is running to mr. krabs to tattle , with spongebob chasing him , is pretty much how it happened in real life . " the end of the episode , where mr. krabs uses more profanity than spongebob and patrick , was also inspired " by the fact that my [ drymon 's ] mother has a sailor mouth herself . "`
- Prediction: `the the,, the the the the the the the the the the the the ", the the the the the @ @ @ @ @ @ @ @ @ @ @ @ " " " " @ @ @ @ @ @ @ @ " " " the @ @ @ @ @ @ @ @ @ @ @`
- Exact match: `0.000`
- Token F1: `0.159`
