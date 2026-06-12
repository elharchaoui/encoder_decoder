# Decoder-Only Evaluation Report

## Run

- Config: `configs/qwen_05b_hotpotqa_3k_seed37.yaml`
- Checkpoint: `runs/qwen_05b_hotpotqa_3k_seed37/best`
- Model: `Qwen/Qwen2-0.5B-Instruct`
- Data source: `hotpotqa`

## Metrics

| Metric | Value |
| --- | ---: |
| `generation_examples` | 256.0000 |
| `exact_match` | 0.0664 |
| `token_f1` | 0.1415 |
| `source_copy_token_f1` | 0.0108 |
| `prediction_source_copy_ratio` | 0.7071 |
| `unique_predictions` | 256.0000 |
| `top_prediction_ratio` | 0.0039 |
| `empty_prediction_ratio` | 0.0000 |
| `avg_prediction_length_tokens` | 3.2461 |
| `num_beams` | 4.0000 |

## Samples

### Sample 1

- Source: `question: on what street would one find the "journal record building" in oklahoma? context: pierre parrant: pierre "pig's eye" parrant, or pierre parent, was the first person of european descent to li`
- Target: `nw 5th`
- Prediction: `12th street, oklahoma city`
- Exact match: `0.000`
- Token F1: `0.000`

### Sample 2

- Source: `question: bally's & paris station is on the monorail that is of what length? context: maharajalela monorail station: maharajalela monorail station (previously merdeka station) is a malaysian elevated `
- Target: `3.9 mi`
- Prediction: `; paris station`
- Exact match: `0.000`
- Token F1: `0.000`

### Sample 3

- Source: `question: name five actors that worked with a german cinematographer? context: fight for fame: fight for fame is a one-hour reality show produced by e! entertainment television, and producers jay jame`
- Target: `george clooney, thekla reuten, violante placido, irina björklund, and paolo bonacelli`
- Prediction: `josef "sepp" allgeier, luis trenker,`
- Exact match: `0.000`
- Token F1: `0.000`

### Sample 4

- Source: `question: emu australia is best known for their unisex style boots that are typically made of what? context: cavalier boots: cavalier boots are a style of boot that were popular in europe between appr`
- Target: `sheepskin`
- Prediction: `beaver-hair felt`
- Exact match: `0.000`
- Token F1: `0.000`

### Sample 5

- Source: `question: what film studio produced both national treasure and the computer wore tennis shoes? context: the computer wore tennis shoes: the computer wore tennis shoes is a 1969 american comedy film st`
- Target: `walt disney pictures`
- Prediction: `walt disney productions`
- Exact match: `0.000`
- Token F1: `0.667`

### Sample 6

- Source: `question: the actor who plays the grown version of emma swan's son, from "once upon a time", also plays what character in the television show "the walking dead"? context: andrew j. west: andrew james `
- Target: `gareth`
- Prediction: `jared s. gilmore`
- Exact match: `0.000`
- Token F1: `0.000`

### Sample 7

- Source: `question: which park is larger of the two, doñana national park or timanfaya national park? context: doñana national park: doñana national park is a natural reserve in andalusia, southern spain, in th`
- Target: `doñana national park`
- Prediction: `timanfaya national park`
- Exact match: `0.000`
- Token F1: `0.667`

### Sample 8

- Source: `question: which of the actress starred in zombie night appeared in the 1998 erotic thriller "wild things"? context: wild things (film): wild things is a 1998 american erotic thriller film directed by `
- Target: `jennifer taylor`
- Prediction: `kevin bacon`
- Exact match: `0.000`
- Token F1: `0.000`
