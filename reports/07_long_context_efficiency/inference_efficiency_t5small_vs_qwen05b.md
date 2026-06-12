# Inference Efficiency: Encoder-Decoder vs Decoder-Only

## Setup

- Encoder-decoder model: `google-t5/t5-small` (60.5M params)
- Decoder-only model: `Qwen/Qwen2-0.5B-Instruct` (494.0M params)
- Answer tokens per query: `8`
- Device: `cuda`

## Crossover Points

The crossover K is the number of queries at which encoder-decoder total latency
becomes lower than decoder-only total latency for the same document.

| Context length | Crossover K | Enc-dec enc (ms) | Enc-dec dec/query (ms) | Dec-only/query (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.06 | 2.38 | 27.15 | 68.95 |
| 128 | 0.04 | 1.77 | 26.23 | 70.4 |
| 256 | 0.05 | 2.25 | 26.41 | 73.32 |
| 384 | 0.06 | 3.19 | 27.85 | 81.25 |

## Latency Comparison (ms)

| Context length | Query count | Enc-dec total (ms) | Dec-only total (ms) | Enc-dec speedup |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 29.53 | 68.95 | 2.33x |
| 64 | 3 | 83.83 | 206.85 | 2.47x |
| 64 | 5 | 138.13 | 344.76 | 2.50x |
| 64 | 10 | 273.87 | 689.52 | 2.52x |
| 64 | 20 | 545.35 | 1379.03 | 2.53x |
| 128 | 1 | 28.0 | 70.4 | 2.51x |
| 128 | 3 | 80.46 | 211.21 | 2.63x |
| 128 | 5 | 132.92 | 352.02 | 2.65x |
| 128 | 10 | 264.06 | 704.04 | 2.67x |
| 128 | 20 | 526.36 | 1408.08 | 2.68x |
| 256 | 1 | 28.66 | 73.32 | 2.56x |
| 256 | 3 | 81.49 | 219.96 | 2.70x |
| 256 | 5 | 134.32 | 366.59 | 2.73x |
| 256 | 10 | 266.38 | 733.19 | 2.75x |
| 256 | 20 | 530.51 | 1466.37 | 2.76x |
| 384 | 1 | 31.04 | 81.25 | 2.62x |
| 384 | 3 | 86.74 | 243.76 | 2.81x |
| 384 | 5 | 142.44 | 406.26 | 2.85x |
| 384 | 10 | 281.7 | 812.52 | 2.88x |
| 384 | 20 | 560.2 | 1625.04 | 2.90x |

## Approximate FLOPs Comparison

| Context length | Query count | Enc-dec GFLOPs | Dec-only GFLOPs | Enc-dec savings |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 3.2 | 41.421 | 92.3% |
| 64 | 3 | 4.717 | 124.264 | 96.2% |
| 64 | 5 | 6.234 | 207.107 | 97.0% |
| 64 | 10 | 10.026 | 414.213 | 97.6% |
| 64 | 20 | 17.611 | 828.427 | 97.9% |
| 128 | 1 | 6.097 | 78.613 | 92.2% |
| 128 | 3 | 8.425 | 235.84 | 96.4% |
| 128 | 5 | 10.754 | 393.066 | 97.3% |
| 128 | 10 | 16.576 | 786.133 | 97.9% |
| 128 | 20 | 28.219 | 1572.266 | 98.2% |
| 256 | 1 | 12.042 | 154.054 | 92.2% |
| 256 | 3 | 15.994 | 462.162 | 96.5% |
| 256 | 5 | 19.946 | 770.271 | 97.4% |
| 256 | 10 | 29.825 | 1540.541 | 98.1% |
| 256 | 20 | 49.585 | 3081.083 | 98.4% |
| 384 | 1 | 18.189 | 230.904 | 92.1% |
| 384 | 3 | 23.764 | 692.713 | 96.6% |
| 384 | 5 | 29.339 | 1154.521 | 97.5% |
| 384 | 10 | 43.277 | 2309.043 | 98.1% |
| 384 | 20 | 71.152 | 4618.085 | 98.5% |

## Interpretation

- At context length `64`: encoder-decoder becomes cheaper after `0.06` queries.
- At context length `128`: encoder-decoder becomes cheaper after `0.04` queries.
- At context length `256`: encoder-decoder becomes cheaper after `0.05` queries.
- At context length `384`: encoder-decoder becomes cheaper after `0.06` queries.

For any production system where the same document is queried more than the crossover K times
(RAG, chatbot over a document, multi-turn QA), encoder-decoder has strictly lower inference cost.