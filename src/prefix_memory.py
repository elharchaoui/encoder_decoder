from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM


@dataclass(frozen=True)
class PrefixMemoryConfig:
    encoder_name: str
    decoder_name: str
    memory_tokens: int = 64
    bridge_heads: int = 8
    bridge_dropout: float = 0.0
    freeze_encoder: bool = True
    freeze_decoder: bool = True
    decoder_trainable_patterns: tuple[str, ...] = ()


class PrefixMemoryCausalLM(nn.Module):
    def __init__(self, config: PrefixMemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        self.decoder = AutoModelForCausalLM.from_pretrained(config.decoder_name)

        encoder_hidden = self.encoder.config.hidden_size
        decoder_hidden = self.decoder.config.hidden_size
        self.memory_queries = nn.Parameter(torch.empty(config.memory_tokens, encoder_hidden))
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=encoder_hidden,
            num_heads=config.bridge_heads,
            dropout=config.bridge_dropout,
            batch_first=True,
        )
        self.memory_norm = nn.LayerNorm(encoder_hidden)
        self.memory_projection = nn.Sequential(
            nn.Linear(encoder_hidden, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.LayerNorm(decoder_hidden),
        )
        self.reset_bridge_parameters()

        if config.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        if config.freeze_decoder:
            for parameter in self.decoder.parameters():
                parameter.requires_grad = False
        for name, parameter in self.decoder.named_parameters():
            if any(pattern in name for pattern in config.decoder_trainable_patterns):
                parameter.requires_grad = True

    def reset_bridge_parameters(self) -> None:
        nn.init.normal_(self.memory_queries, mean=0.0, std=0.02)

    @property
    def decoder_embedding(self) -> nn.Module:
        return self.decoder.get_input_embeddings()

    def encode_memory(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_outputs = self.encoder(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
        )
        encoder_states = encoder_outputs.last_hidden_state
        batch_size = encoder_states.size(0)
        queries = self.memory_queries.unsqueeze(0).expand(batch_size, -1, -1)
        key_padding_mask = source_attention_mask == 0
        memory, _ = self.memory_attention(
            query=queries,
            key=encoder_states,
            value=encoder_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        memory = self.memory_norm(memory)
        return self.memory_projection(memory)

    def forward(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        memory_embeds = self.encode_memory(source_input_ids, source_attention_mask)
        target_embeds = self.decoder_embedding(target_input_ids)
        inputs_embeds = torch.cat([memory_embeds, target_embeds], dim=1)
        memory_attention_mask = torch.ones(
            memory_embeds.shape[:2],
            dtype=target_attention_mask.dtype,
            device=target_attention_mask.device,
        )
        attention_mask = torch.cat([memory_attention_mask, target_attention_mask], dim=1)
        memory_labels = torch.full(
            memory_embeds.shape[:2],
            fill_value=-100,
            dtype=labels.dtype,
            device=labels.device,
        )
        decoder_labels = torch.cat([memory_labels, labels], dim=1)
        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=decoder_labels,
        )

    @torch.no_grad()
    def generate(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int | None,
        max_new_tokens: int,
        min_new_tokens: int = 0,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.eval()
        memory_embeds = self.encode_memory(source_input_ids, source_attention_mask)
        batch_size = memory_embeds.size(0)
        memory_attention_mask = torch.ones(
            memory_embeds.shape[:2],
            dtype=source_attention_mask.dtype,
            device=source_attention_mask.device,
        )
        if prompt_input_ids is not None:
            prompt_embeds = self.decoder_embedding(prompt_input_ids)
            memory_embeds = torch.cat([memory_embeds, prompt_embeds], dim=1)
            if prompt_attention_mask is None:
                prompt_attention_mask = torch.ones_like(prompt_input_ids)
            attention_mask = torch.cat([memory_attention_mask, prompt_attention_mask], dim=1)
        else:
            attention_mask = memory_attention_mask
        outputs = self.decoder(
            inputs_embeds=memory_embeds,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        if eos_token_id is not None and min_new_tokens > 0:
            logits[:, eos_token_id] = -torch.inf
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = [next_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source_input_ids.device)

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None:
                finished |= next_token.squeeze(1).eq(eos_token_id)
                if bool(finished.all()):
                    break
            outputs = self.decoder(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            if eos_token_id is not None and len(generated) < min_new_tokens:
                logits[:, eos_token_id] = -torch.inf
            next_token = logits.argmax(dim=-1, keepdim=True)
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )
            generated.append(next_token)

        if not generated:
            return torch.full(
                (batch_size, 1),
                fill_value=bos_token_id,
                dtype=torch.long,
                device=source_input_ids.device,
            )
        return torch.cat(generated, dim=1)
