from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

from src.corruption import CorruptionConfig, corrupt_text, corruption_config_from_dict, normalize_text


DEFAULT_SENTENCES = [
    "The model generates one token at a time.",
    "Paris is the capital of France.",
    "The cat sat on the mat.",
    "A frozen encoder can provide reusable semantic memory.",
    "The decoder attends to previous target tokens and encoder states.",
    "Training uses teacher forcing with shifted target tokens.",
    "The source encoder should run exactly once during generation.",
    "A small autoregressive decoder predicts the next token.",
    "Synthetic denoising data is useful for the first experiment.",
    "The validation loop measures reconstruction quality.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning systems should be tested with simple baselines.",
    "The experiment compares generated text against the clean target.",
    "Cross attention lets the decoder inspect the encoded source.",
    "Padding labels are ignored by the language modeling loss.",
]


@dataclass(frozen=True)
class TextPair:
    source: str
    target: str


@dataclass(frozen=True)
class SpanTargetConfig:
    sentinel_token: str = "[unused1]"
    min_span_length: int = 2
    max_span_length: int = 8
    min_left_words: int = 2
    min_right_words: int = 2


def span_target_config_from_dict(values: dict | None) -> SpanTargetConfig:
    if not values:
        return SpanTargetConfig()
    allowed = set(SpanTargetConfig.__dataclass_fields__)
    filtered = {key: value for key, value in values.items() if key in allowed}
    return SpanTargetConfig(**filtered)


def build_synthetic_pairs(
    count: int,
    seed: int,
    mask_token: str,
    corruption_config: CorruptionConfig | None = None,
) -> list[TextPair]:
    rng = random.Random(seed)
    pairs: list[TextPair] = []
    for idx in range(count):
        clean = DEFAULT_SENTENCES[idx % len(DEFAULT_SENTENCES)]
        target = normalize_text(clean)
        source = corrupt_text(
            target,
            rng,
            mask_token=mask_token,
            config=corruption_config,
        )
        pairs.append(TextPair(source=source, target=target))
    rng.shuffle(pairs)
    return pairs


def _usable_wikitext_line(text: str) -> bool:
    text = text.strip()
    if len(text) < 40:
        return False
    if text.startswith("=") and text.endswith("="):
        return False
    if not re.search(r"[a-zA-Z]", text):
        return False
    return True


def _split_sentence_like_chunks(text: str, min_words: int, max_words: int) -> list[str]:
    text = normalize_text(text)
    rough_sentences = re.split(r"(?<=[.!?])\s+|;\s+|\s+-\s+", text)
    chunks: list[str] = []
    for sentence in rough_sentences:
        sentence = normalize_text(sentence)
        words = sentence.split()
        if len(words) < min_words:
            continue
        for start in range(0, len(words), max_words):
            chunk_words = words[start : start + max_words]
            if len(chunk_words) >= min_words:
                chunks.append(" ".join(chunk_words))
    return chunks


def make_span_target_pair(
    text: str,
    rng: random.Random,
    config: SpanTargetConfig | None = None,
) -> TextPair | None:
    config = config or SpanTargetConfig()
    words = normalize_text(text).split()
    min_required = config.min_left_words + config.min_span_length + config.min_right_words
    if len(words) < min_required:
        return None
    max_span = min(
        config.max_span_length,
        len(words) - config.min_left_words - config.min_right_words,
    )
    if max_span < config.min_span_length:
        return None
    span_len = rng.randint(config.min_span_length, max_span)
    start_min = config.min_left_words
    start_max = len(words) - config.min_right_words - span_len
    if start_max < start_min:
        return None
    start = rng.randint(start_min, start_max)
    end = start + span_len
    source_words = words[:start] + [config.sentinel_token] + words[end:]
    target_words = words[start:end]
    return TextPair(source=" ".join(source_words), target=" ".join(target_words))


def build_wikitext_pairs(
    count: int,
    seed: int,
    mask_token: str,
    split: str,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    corruption_config: CorruptionConfig | None = None,
    objective: str = "full_reconstruction",
    span_target_config: SpanTargetConfig | None = None,
    min_words: int = 8,
    max_words: int = 48,
) -> list[TextPair]:
    from datasets import load_dataset

    rng = random.Random(seed)
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    clean_chunks: list[str] = []
    for row in dataset:
        text = row.get("text", "")
        if not _usable_wikitext_line(text):
            continue
        clean_chunks.extend(
            _split_sentence_like_chunks(
                text,
                min_words=min_words,
                max_words=max_words,
            )
        )
    if not clean_chunks:
        raise RuntimeError(f"No usable text lines found in {dataset_name}/{dataset_config}:{split}")
    rng.shuffle(clean_chunks)
    selected = clean_chunks[:count]
    if len(selected) < count:
        raise RuntimeError(
            f"Requested {count} examples from {split}, but only found {len(selected)} usable lines."
        )
    pairs: list[TextPair] = []
    for target in selected:
        if objective == "span_target":
            pair = make_span_target_pair(
                target,
                rng,
                config=span_target_config,
            )
            if pair is None:
                continue
            pairs.append(pair)
            continue
        if objective == "full_reconstruction":
            pairs.append(
                TextPair(
                    source=corrupt_text(
                        target,
                        rng,
                        mask_token=mask_token,
                        config=corruption_config,
                    ),
                    target=target,
                )
            )
            continue
        raise ValueError(f"Unsupported data.objective: {objective}")
    if len(pairs) < count:
        raise RuntimeError(
            f"Requested {count} examples from {split}, but only built {len(pairs)} pairs."
        )
    return pairs[:count]


def build_pairs_from_config(cfg: dict, seed: int, mask_token: str, split: str) -> list[TextPair]:
    data_cfg = cfg["data"]
    source = data_cfg.get("source", "synthetic")
    objective = data_cfg.get("objective", "full_reconstruction")
    corruption_config = corruption_config_from_dict(data_cfg.get("corruption"))
    span_target_config = span_target_config_from_dict(data_cfg.get("span_target"))
    count_key = "train_examples" if split == "train" else "validation_examples"
    count = int(data_cfg[count_key])
    if source == "synthetic":
        split_seed = seed if split == "train" else seed + 1
        return build_synthetic_pairs(
            count=count,
            seed=split_seed,
            mask_token=mask_token,
            corruption_config=corruption_config,
        )
    if source == "wikitext":
        hf_split = data_cfg.get("train_split" if split == "train" else "validation_split", split)
        return build_wikitext_pairs(
            count=count,
            seed=seed if split == "train" else seed + 1,
            mask_token=mask_token,
            split=hf_split,
            dataset_name=data_cfg.get("dataset_name", "Salesforce/wikitext"),
            dataset_config=data_cfg.get("dataset_config", "wikitext-2-raw-v1"),
            corruption_config=corruption_config,
            objective=objective,
            span_target_config=span_target_config,
            min_words=int(data_cfg.get("min_words", 8)),
            max_words=int(data_cfg.get("max_words", 48)),
        )
    raise ValueError(f"Unsupported data.source: {source}")


class DenoisingDataset(Dataset):
    def __init__(
        self,
        pairs: Iterable[TextPair],
        tokenizer,
        source_max_length: int,
        target_max_length: int,
    ) -> None:
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        source = self.tokenizer(
            pair.source,
            max_length=self.source_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer(
            pair.target,
            max_length=self.target_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target_ids = target["input_ids"].squeeze(0)
        decoder_input_ids = target_ids[:-1].clone()
        labels = target_ids[1:].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "source_text": pair.source,
            "target_text": pair.target,
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
