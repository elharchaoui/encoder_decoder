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


def build_wikitext_pairs(
    count: int,
    seed: int,
    mask_token: str,
    split: str,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    corruption_config: CorruptionConfig | None = None,
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
    return [
        TextPair(
            source=corrupt_text(
                target,
                rng,
                mask_token=mask_token,
                config=corruption_config,
            ),
            target=target,
        )
        for target in selected
    ]


def build_pairs_from_config(cfg: dict, seed: int, mask_token: str, split: str) -> list[TextPair]:
    data_cfg = cfg["data"]
    source = data_cfg.get("source", "synthetic")
    corruption_config = corruption_config_from_dict(data_cfg.get("corruption"))
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
