# Decoder-Only Evaluation Report

## Run

- Config: `configs/gpt2_small_squad_ctx320_eval.yaml`
- Checkpoint: `runs/gpt2_small_squad_3k_seed37/best`
- Model: `gpt2`
- Data source: `qa`

## Metrics

| Metric | Value |
| --- | ---: |
| `generation_examples` | 512.0000 |
| `exact_match` | 0.1465 |
| `token_f1` | 0.2633 |
| `source_copy_token_f1` | 0.0427 |
| `prediction_source_copy_ratio` | 0.8607 |
| `unique_predictions` | 499.0000 |
| `top_prediction_ratio` | 0.0059 |
| `empty_prediction_ratio` | 0.0000 |
| `avg_prediction_length_tokens` | 8.3594 |
| `num_beams` | 4.0000 |

## Samples

### Sample 1

- Source: `question: what type of vote must the parliament have to either block or suggest changes to the commission's proposals? context: to make new legislation, tfeu article 294 defines the "ordinary legislat`
- Target: `a majority`
- Prediction: `can agree at any stage, a "conciliation committee" is convened, representing meps, ministers and the commission to try and get agreement on a joint`
- Exact match: `0.000`
- Token F1: `0.077`

### Sample 2

- Source: `question: "bairn" and "hyem" have origins from what culture? context: "bairn" and "hyem", meaning "child" and "home", respectively, are examples of geordie words with origins in scandinavia; barn and `
- Target: `scandinavia`
- Prediction: `scandinavia`
- Exact match: `1.000`
- Token F1: `1.000`

### Sample 3

- Source: `question: what did the the europeans think the peoples in the tropics were in need of? context: according to geographic scholars under colonizing empires, the world could be split into climatic zones.`
- Target: `guidance`
- Prediction: `the people of these climates were believed to be in need of guidance and intervention from the european empire to aid in the governing of a more evolved social structure`
- Exact match: `0.000`
- Token F1: `0.065`

### Sample 4

- Source: `question: what do students learn about in confirmation and membership preparation classes? context: unlike confirmation and profession of faith, baptism is a sacrament in the umc. the book of discipli`
- Target: `church and the methodist-christian theological tradition`
- Prediction: `church and the methodist-christian theological tradition in order to profess their ultimate faith in christ.`
- Exact match: `0.000`
- Token F1: `0.571`

### Sample 5

- Source: `question: what are the two major thoroughfares of sunnyside? context: the neighborhood of sunnyside is on fresno's far southeast side, bounded by chestnut avenue to the west. its major thoroughfares a`
- Target: `kings canyon avenue and clovis avenue`
- Prediction: `king canyon avenue and clovis avenue`
- Exact match: `0.000`
- Token F1: `0.833`

### Sample 6

- Source: `question: what does the zachęta national gallery of art organize exhibitions of art from? context: the 17th century royal ujazdów castle currently houses centre for contemporary art, with some permane`
- Target: `polish and international artists`
- Prediction: `the 17th century royal ujazdów castle currently houses centre for contemporary art, with some permanent and temporary exhibitions, concerts, shows and creative workshops`
- Exact match: `0.000`
- Token F1: `0.074`

### Sample 7

- Source: `question: what position did luther have in wittenberg? context: on 19 october 1512, he was awarded his doctor of theology and, on 21 october 1512, was received into the senate of the theological facul`
- Target: `doctor in bible`
- Prediction: `doctor in bible`
- Exact match: `1.000`
- Token F1: `1.000`

### Sample 8

- Source: `question: who kicked ethelred out? context: the normans were in contact with england from an early date. not only were their original viking brethren still ravaging the english coasts, they occupied m`
- Target: `sweyn forkbeard`
- Prediction: `emma, sister of duke richard ii of normandy`
- Exact match: `0.000`
- Token F1: `0.000`
