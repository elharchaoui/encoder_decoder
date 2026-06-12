# Decoder-Only Evaluation Report

## Run

- Config: `configs/gpt2_large_squad_30k_seed37_eval512.yaml`
- Checkpoint: `runs/gpt2_large_squad_30k_seed37/best`
- Model: `gpt2-large`
- Data source: `qa`

## Metrics

| Metric | Value |
| --- | ---: |
| `generation_examples` | 512.0000 |
| `exact_match` | 0.3516 |
| `token_f1` | 0.5041 |
| `source_copy_token_f1` | 0.0427 |
| `prediction_source_copy_ratio` | 0.8078 |
| `unique_predictions` | 496.0000 |
| `top_prediction_ratio` | 0.0059 |
| `empty_prediction_ratio` | 0.0000 |
| `avg_prediction_length_tokens` | 3.6543 |
| `num_beams` | 4.0000 |
| `bertscore_precision` | 0.5224 |
| `bertscore_recall` | 0.5304 |
| `bertscore_f1` | 0.5250 |

## Samples

### Sample 1

- Source: `question: what type of vote must the parliament have to either block or suggest changes to the commission's proposals? context: to make new legislation, tfeu article 294 defines the "ordinary legislat`
- Target: `a majority`
- Prediction: `legislation can be blocked by a majority in parliament, a minority in the council, and a majority in the commission`
- Exact match: `0.000`
- Token F1: `0.182`

### Sample 2

- Source: `question: "bairn" and "hyem" have origins from what culture? context: "bairn" and "hyem", meaning "child" and "home", respectively, are examples of geordie words with origins in scandinavia; barn and `
- Target: `scandinavia`
- Prediction: `scandinavia`
- Exact match: `1.000`
- Token F1: `1.000`

### Sample 3

- Source: `question: what did the the europeans think the peoples in the tropics were in need of? context: according to geographic scholars under colonizing empires, the world could be split into climatic zones.`
- Target: `guidance`
- Prediction: `guidance and intervention`
- Exact match: `0.000`
- Token F1: `0.500`

### Sample 4

- Source: `question: what do students learn about in confirmation and membership preparation classes? context: unlike confirmation and profession of faith, baptism is a sacrament in the umc. the book of discipli`
- Target: `church and the methodist-christian theological tradition`
- Prediction: `church and the methodist-christian theological tradition`
- Exact match: `1.000`
- Token F1: `1.000`

### Sample 5

- Source: `question: what are the two major thoroughfares of sunnyside? context: the neighborhood of sunnyside is on fresno's far southeast side, bounded by chestnut avenue to the west. its major thoroughfares a`
- Target: `kings canyon avenue and clovis avenue`
- Prediction: `clovis avenue`
- Exact match: `0.000`
- Token F1: `0.500`

### Sample 6

- Source: `question: what does the zachęta national gallery of art organize exhibitions of art from? context: the 17th century royal ujazdów castle currently houses centre for contemporary art, with some permane`
- Target: `polish and international artists`
- Prediction: `modern art`
- Exact match: `0.000`
- Token F1: `0.000`

### Sample 7

- Source: `question: what position did luther have in wittenberg? context: on 19 october 1512, he was awarded his doctor of theology and, on 21 october 1512, was received into the senate of the theological facul`
- Target: `doctor in bible`
- Prediction: `doctor in bible`
- Exact match: `1.000`
- Token F1: `1.000`

### Sample 8

- Source: `question: who kicked ethelred out? context: the normans were in contact with england from an early date. not only were their original viking brethren still ravaging the english coasts, they occupied m`
- Target: `sweyn forkbeard`
- Prediction: `sweyn forkbeard`
- Exact match: `1.000`
- Token F1: `1.000`
