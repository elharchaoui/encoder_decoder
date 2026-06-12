# Prefix-Memory Evaluation Report

## Run

- Config: `configs/prefix_memory_qwen05_adapter_wikitext_span_short_debug.yaml`
- Checkpoint: `runs/prefix_memory_qwen05_adapter_wikitext_span_short_debug/best`
- Encoder: `thenlper/gte-small`
- Decoder baseline: `Qwen/Qwen2-0.5B-Instruct`
- Memory tokens: `64`
- Data objective: `span_target`

## Metrics

| Metric | Value |
| --- | ---: |
| `validation_loss` | 9.9577 |
| `perplexity` | 21113.26 |
| `generation_examples` | 64.0000 |
| `prefix_exact_match` | 0.0000 |
| `prefix_token_f1` | 0.0073 |
| `source_copy_token_f1` | 0.0374 |
| `prefix_prediction_source_copy_ratio` | 0.0361 |
| `decoder_baseline_exact_match` | 0.0000 |
| `decoder_baseline_token_f1` | 0.0464 |
| `decoder_baseline_prediction_source_copy_ratio` | 0.3565 |
| `prefix_gain_over_source_copy` | -0.0301 |
| `prefix_gap_to_decoder_baseline` | 0.0391 |
| `prefix_target_len_1_token_f1` | 0.0000 |
| `target_len_1_examples` | 25.0000 |
| `prefix_target_len_2_3_token_f1` | 0.0120 |
| `target_len_2_3_examples` | 39.0000 |

## Samples

### Sample 1

- Source: `the freeway portion of route 29 ends at the intersection with lee avenue and it continues northwest <missing> delaware river as a four @-@ lane divided highway .`
- Target: `along the`
- Prefix-memory prediction: `1.1000000`
- Decoder-only baseline: `The freeway portion of route 29 ends at`
- Prefix-memory token F1: `0.000`
- Decoder baseline token F1: `0.200`

### Sample 2

- Source: `in the <missing> 6th century , caracol seems to have allied with calakmul and defeated tikal , closing the early classic .`
- Target: `mid`
- Prefix-memory prediction: `100000000`
- Decoder-only baseline: `in the 6th century, caracol`
- Prefix-memory token F1: `0.000`
- Decoder baseline token F1: `0.000`

### Sample 3

- Source: `in later choruses , bono sings " blown <missing> " with the same melody , stretching the same note even longer .`
- Target: `by the wind`
- Prefix-memory prediction: `1111111111`
- Decoder-only baseline: `" blown" The missing span is "blown`
- Prefix-memory token F1: `0.000`
- Decoder baseline token F1: `0.200`

### Sample 4

- Source: `rand said about the <missing> general that it " plays like a dream "`
- Target: `gameplay in`
- Prefix-memory prediction: `100000000`
- Decoder-only baseline: `"that" The missing span in the text is`
- Prefix-memory token F1: `0.000`
- Decoder baseline token F1: `0.200`

### Sample 5

- Source: `the timber used was 1 1 ⁄ 2 inches ( 38 mm ) thick and nine laminations were required at the centre , where the stock passes through <missing> of the windshaft .`
- Target: `the poll end`
- Prefix-memory prediction: `1. What is the main idea of the`
- Decoder-only baseline: `The missing span is not provided in the given text`
- Prefix-memory token F1: `0.182`
- Decoder baseline token F1: `0.154`

### Sample 6

- Source: `the average annual rainfall at tikal is 1 @,@ 945 millimetres <missing> 76 @.@ 6 in ) .`
- Target: `(`
- Prefix-memory prediction: `100000000`
- Decoder-only baseline: `"the average annual rainfall at tikal is`
- Prefix-memory token F1: `0.000`
- Decoder baseline token F1: `0.000`

### Sample 7

- Source: `gauthier travelled to france , where she received private <missing> lessons from auguste @-@ jean dubulle of the paris conservatory .`
- Target: `voice`
- Prefix-memory prediction: `100000000`
- Decoder-only baseline: `"gauthier travelled to France, where she`
- Prefix-memory token F1: `0.000`
- Decoder baseline token F1: `0.000`

### Sample 8

- Source: `they purchased five guns in texas , but encountered difficulty purchasing handguns in texas <missing> out @-@ of @-@ state identification and traveled to new mexico instead .`
- Target: `with`
- Prefix-memory prediction: `100000000`
- Decoder-only baseline: `"but encountered difficulty purchasing handguns in Texas" The`
- Prefix-memory token F1: `0.000`
- Decoder baseline token F1: `0.000`
