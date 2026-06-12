from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

from src.data import TextPair


@dataclass(frozen=True)
class Seq2SeqBatchConfig:
    source_max_length: int
    target_max_length: int


class Seq2SeqDenoisingDataset(Dataset):
    def __init__(
        self,
        pairs: Iterable[TextPair],
        tokenizer,
        config: Seq2SeqBatchConfig,
        prefix: str = "",
    ) -> None:
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.config = config
        self.prefix = prefix

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        source_text = f"{self.prefix}{pair.source}"
        source = self.tokenizer(
            source_text,
            max_length=self.config.source_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer(
            pair.target,
            max_length=self.config.target_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = target["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "source_text": pair.source,
            "target_text": pair.target,
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
        }
