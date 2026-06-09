User :
I want to do an experiment, pick an encoder only and attach to it a decoder that will run recursively to generate one token at the time, without looping back to encoder.
AI :
Yes — what you described is essentially a **sequence-to-sequence encoder-decoder model**, where the **encoder runs once**, and the **decoder generates autoregressively**.

The architecture would be:

```text
Input text
   ↓
Encoder-only model, e.g. BERT / RoBERTa / DeBERTa
   ↓
Fixed encoder hidden states H
   ↓
Causal decoder
   ↓
Generate token 1
   ↓
Generate token 2
   ↓
Generate token 3
   ↓
...
```

At generation time:

```text
H = Encoder(x)                       # run once

y₀ = <BOS>

for t in 1..T:
    yₜ = Decoder(y₀, ..., yₜ₋₁, H)   # no re-encoding
```

So mathematically:

```text
p(y | x) = ∏ₜ p(yₜ | y₁...yₜ₋₁, Encoder(x))
```

## Best encoder choice for a first experiment

I would pick **BERT-base** first.

Why:

* simple and well-supported;
* WordPiece tokenizer works fine for generation experiments;
* Hugging Face already supports turning BERT-like models into encoder-decoder systems;
* small enough to debug quickly;
* strong enough to produce meaningful encoder representations.

A good first version:

```text
Encoder: bert-base-uncased
Decoder: small causal Transformer decoder
```

or, for easier initialization:

```text
Encoder: bert-base-uncased
Decoder: bert-base-uncased configured as a causal decoder
```

This second version is basically a **BERT2BERT** setup.

## Important detail: the decoder needs cross-attention

The decoder should not just receive the encoder output once at the beginning. It should have **cross-attention** layers:

```text
Decoder layer:
    causal self-attention over generated tokens
    cross-attention over fixed encoder states
    feed-forward layer
```

So each generated token can attend to the full encoded input, but the encoder itself is not rerun.

```text
Generated tokens so far ── causal self-attention ┐
                                                  ↓
Encoder hidden states ── cross-attention ─────→ decoder output
```

## Minimal Hugging Face prototype

```python
from transformers import EncoderDecoderModel, AutoTokenizer

encoder_name = "bert-base-uncased"
decoder_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(encoder_name)

model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_name,
    decoder_name
)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.config.max_length = 64
model.config.num_beams = 4
```

Training example:

```python
inputs = tokenizer(
    source_texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

labels = tokenizer(
    target_texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
).input_ids

labels[labels == tokenizer.pad_token_id] = -100

outputs = model(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    labels=labels
)

loss = outputs.loss
loss.backward()
```

Generation:

```python
inputs = tokenizer(
    ["The source text goes here."],
    return_tensors="pt"
)

generated = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_length=64
)

print(tokenizer.decode(generated[0], skip_special_tokens=True))
```

## To ensure the encoder is not looped

In a manual implementation, you would do:

```python
encoder_outputs = model.encoder(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask
)

generated = model.generate(
    encoder_outputs=encoder_outputs,
    attention_mask=inputs.attention_mask,
    max_length=64
)
```

Conceptually, this does:

```text
encoder: once
decoder: many times
```

For efficiency, you would also cache decoder key/value states during generation:

```text
Step 1: decoder computes KV cache
Step 2: reuse past KV cache
Step 3: reuse past KV cache
...
```

The encoder memory stays fixed.

## Training options

You have three sensible experiment variants:

| Variant                    | Description                                                                                                            |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Frozen encoder**         | Keep BERT fixed, train only the decoder and cross-attention. Good for testing whether BERT representations are enough. |
| **Fine-tuned encoder**     | Train encoder + decoder end-to-end. Usually much better.                                                               |
| **LoRA / adapter encoder** | Middle ground: mostly freeze BERT but allow lightweight adaptation.                                                    |

I would start with:

```text
Frozen BERT encoder + randomly initialized 4–6 layer decoder
```

Then compare against:

```text
Fine-tuned BERT encoder + decoder
```

## Main caveat

An encoder-only model like BERT was not trained to support generation directly. Its representations are excellent for understanding, but the decoder has to learn how to turn those representations into fluent autoregressive text.

So the system will need proper seq2seq training data, for example:

```text
input text  →  target text
question    →  answer
document    →  summary
corrupted sentence → clean sentence
source sentence → paraphrase
```

For a clean first experiment, I would use a **denoising autoencoder task**:

```text
Input:  "The cat [MASK] on the mat."
Target: "The cat sat on the mat."
```

or:

```text
Input:  corrupted / shuffled / truncated text
Target: original text
```

That aligns naturally with BERT-style representations while still testing autoregressive decoding.

User :
Is this has been tested before 
AI:
Yes. This idea has been tested before, and fairly extensively.

Your proposal is closest to these lines of work:

| Setup                                                       | Has it been tested? | Notes                                                                                                                                                                     |
| ----------------------------------------------------------- | ------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **BERT encoder + randomly initialized Transformer decoder** |                 Yes | Used for abstractive summarization in **BERTSUMABS**. The encoder is pretrained BERT/BERTSUM, while the decoder is trained from scratch. ([arXiv][1])                     |
| **BERT encoder + BERT initialized as decoder**              |                 Yes | Often called **BERT2BERT**. BERT weights initialize both sides, but the decoder is modified to be causal and to use cross-attention. ([aclanthology.org][2])              |
| **BERT/RoBERTa encoder + GPT-2 decoder**                    |                 Yes | Tested as warm-started encoder-decoder models for generation tasks. Rothe et al. studied combinations using BERT, GPT-2, and RoBERTa checkpoints. ([aclanthology.org][2]) |
| **General Hugging Face `EncoderDecoderModel`**              |                 Yes | Hugging Face directly supports initializing a seq2seq model from a pretrained encoder and a pretrained autoregressive decoder. ([huggingface.co][3])                      |
| **BertGeneration**                                          |                 Yes | A Hugging Face model family adapting BERT checkpoints for seq2seq generation with an encoder-decoder setup. ([GitHub][4])                                                 |

So the broad idea is **not new**: using an encoder-only model as the source encoder, running it once, then using an autoregressive decoder to generate token by token is a standard tested architecture.

The closest paper to your exact experiment is probably **“Leveraging Pre-trained Checkpoints for Sequence Generation Tasks”** by Rothe, Narayan, and Severyn. They built Transformer encoder-decoder models compatible with pretrained **BERT, GPT-2, and RoBERTa** checkpoints, and evaluated them on machine translation, summarization, sentence splitting, and sentence fusion. ([aclanthology.org][2])

Another close one is **“Text Summarization with Pretrained Encoders”** by Liu and Lapata. For abstractive summarization, they used a pretrained BERT-based encoder with a Transformer decoder, and specifically discussed the training mismatch between a pretrained encoder and a decoder trained from scratch. ([arXiv][1])

The part that could still be novel is **your exact variant**, for example:

```text
ModernBERT / DeBERTa / E5 / BGE encoder
+
small custom recursive decoder
+
frozen encoder
+
no encoder feedback loop
+
specific task or training objective
```

Especially if your decoder is not a normal Transformer decoder, but something more unusual: recurrent state-space decoder, Mamba-like decoder, diffusion decoder, small recurrent decoder, or a decoder trained to exploit frozen embedding-model representations.

A good experimental framing would be:

```text
Question:
Can a frozen encoder-only LLM representation act as a reusable semantic memory
for a lightweight autoregressive decoder?
```

Baselines to compare against:

```text
1. BERT2BERT
2. BERT encoder + random Transformer decoder
3. BERT encoder + GPT-2 decoder
4. T5-small or BART-base
5. Your encoder + custom recursive decoder
```

So: **yes, the core idea has been tested**, but there is still room to test whether a **modern encoder-only model plus a lightweight recursive decoder** can be competitive or efficient.

[1]: https://arxiv.org/abs/1908.08345?utm_source=chatgpt.com "Text Summarization with Pretrained Encoders"
[2]: https://aclanthology.org/2020.tacl-1.18/?utm_source=chatgpt.com "Leveraging Pre-trained Checkpoints for Sequence ..."
[3]: https://huggingface.co/docs/transformers/en/model_doc/encoder-decoder?utm_source=chatgpt.com "Encoder Decoder Models"
[4]: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/bert-generation.md?utm_source=chatgpt.com "bert-generation.md - huggingface/transformers"
