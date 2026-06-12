from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

from src.data import TextPair


@dataclass(frozen=True)
class PrefixMemoryBatchConfig:
    source_max_length: int
    target_max_length: int


class PrefixMemoryDataset(Dataset):
    def __init__(
        self,
        pairs: Iterable[TextPair],
        encoder_tokenizer,
        decoder_tokenizer,
        config: PrefixMemoryBatchConfig,
        source_prefix: str = "",
        decoder_prefix: str = "",
    ) -> None:
        self.pairs = list(pairs)
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.config = config
        self.source_prefix = source_prefix
        self.decoder_prefix = decoder_prefix

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        source = self.encoder_tokenizer(
            f"{self.source_prefix}{pair.source}",
            max_length=self.config.source_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        prefix_ids = self.decoder_tokenizer(
            self.decoder_prefix,
            add_special_tokens=False,
        )["input_ids"]
        target_text = pair.target
        if self.decoder_tokenizer.eos_token:
            target_text = f"{target_text}{self.decoder_tokenizer.eos_token}"
        target_ids = self.decoder_tokenizer(
            target_text,
            add_special_tokens=False,
        )["input_ids"]
        input_ids = (prefix_ids + target_ids)[: self.config.target_max_length]
        label_ids = ([-100] * len(prefix_ids) + target_ids)[: self.config.target_max_length]
        attention = [1] * len(input_ids)
        pad_length = self.config.target_max_length - len(input_ids)
        if pad_length > 0:
            input_ids.extend([self.decoder_tokenizer.pad_token_id] * pad_length)
            label_ids.extend([-100] * pad_length)
            attention.extend([0] * pad_length)
        return {
            "source_text": pair.source,
            "target_text": pair.target,
            "source_input_ids": source["input_ids"].squeeze(0),
            "source_attention_mask": source["attention_mask"].squeeze(0),
            "target_input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }
