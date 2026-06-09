from __future__ import annotations

import torch
from torch import nn

from src.models import FrozenEncoderAutoregressiveDecoder, FrozenEncoderDecoderConfig


class TinyEncoderOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class TinyEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, input_ids, attention_mask):
        return TinyEncoderOutput(self.embedding(input_ids))


def test_generation_calls_encoder_once(monkeypatch):
    class TinyAutoConfig:
        hidden_size = 16

    monkeypatch.setattr("src.models.AutoConfig.from_pretrained", lambda _: TinyAutoConfig())
    monkeypatch.setattr("src.models.AutoModel.from_pretrained", lambda _: TinyEncoder(16))

    model = FrozenEncoderAutoregressiveDecoder(
        FrozenEncoderDecoderConfig(
            encoder_name="tiny",
            vocab_size=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=32,
            dropout=0.0,
        )
    )
    input_ids = torch.tensor([[4, 5, 6, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])

    _ = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=5)

    assert model.encoder_call_count == 1
