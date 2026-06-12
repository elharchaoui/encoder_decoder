from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


@dataclass(frozen=True)
class FrozenEncoderDecoderConfig:
    encoder_name: str
    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    decoder_layers: int = 4
    decoder_heads: int = 8
    decoder_ffn_dim: int = 3072
    dropout: float = 0.1
    init_decoder_embeddings_from_encoder: bool = True
    init_decoder_layers_from_encoder: bool = False
    tie_token_embeddings: bool = True
    use_cross_attention: bool = True


class FrozenEncoderAutoregressiveDecoder(nn.Module):
    def __init__(self, config: FrozenEncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        encoder_config = AutoConfig.from_pretrained(config.encoder_name)
        self.hidden_size = encoder_config.hidden_size
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            self.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embedding = nn.Embedding(512, self.hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, config.vocab_size, bias=False)
        if config.init_decoder_embeddings_from_encoder:
            self._init_decoder_embeddings_from_encoder()
        if config.init_decoder_layers_from_encoder:
            self._init_decoder_layers_from_encoder()
        if config.tie_token_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.encoder_call_count = 0

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _init_decoder_embeddings_from_encoder(self) -> None:
        if not hasattr(self.encoder, "get_input_embeddings"):
            return
        encoder_embeddings = self.encoder.get_input_embeddings()
        if encoder_embeddings is None:
            return
        source_weight = encoder_embeddings.weight.detach()
        if source_weight.shape != self.token_embedding.weight.shape:
            rows = min(source_weight.size(0), self.token_embedding.weight.size(0))
            cols = min(source_weight.size(1), self.token_embedding.weight.size(1))
            with torch.no_grad():
                self.token_embedding.weight[:rows, :cols].copy_(source_weight[:rows, :cols])
            return
        with torch.no_grad():
            self.token_embedding.weight.copy_(source_weight)

    def _copy_if_same_shape(self, target: torch.Tensor, source: torch.Tensor) -> bool:
        if target.shape != source.shape:
            return False
        with torch.no_grad():
            target.copy_(source.detach())
        return True

    def _init_decoder_layers_from_encoder(self) -> None:
        encoder_layers = getattr(getattr(self.encoder, "encoder", None), "layer", None)
        if encoder_layers is None:
            return
        for decoder_layer_index, decoder_layer in enumerate(self.decoder.layers):
            encoder_layer = encoder_layers[decoder_layer_index % len(encoder_layers)]
            attention = encoder_layer.attention
            intermediate = encoder_layer.intermediate
            output = encoder_layer.output

            self._copy_if_same_shape(
                decoder_layer.self_attn.in_proj_weight[: self.hidden_size],
                attention.self.query.weight,
            )
            self._copy_if_same_shape(
                decoder_layer.self_attn.in_proj_weight[self.hidden_size : 2 * self.hidden_size],
                attention.self.key.weight,
            )
            self._copy_if_same_shape(
                decoder_layer.self_attn.in_proj_weight[2 * self.hidden_size :],
                attention.self.value.weight,
            )
            self._copy_if_same_shape(
                decoder_layer.self_attn.in_proj_bias[: self.hidden_size],
                attention.self.query.bias,
            )
            self._copy_if_same_shape(
                decoder_layer.self_attn.in_proj_bias[self.hidden_size : 2 * self.hidden_size],
                attention.self.key.bias,
            )
            self._copy_if_same_shape(
                decoder_layer.self_attn.in_proj_bias[2 * self.hidden_size :],
                attention.self.value.bias,
            )
            self._copy_if_same_shape(
                decoder_layer.self_attn.out_proj.weight,
                attention.output.dense.weight,
            )
            self._copy_if_same_shape(
                decoder_layer.self_attn.out_proj.bias,
                attention.output.dense.bias,
            )
            self._copy_if_same_shape(decoder_layer.linear1.weight, intermediate.dense.weight)
            self._copy_if_same_shape(decoder_layer.linear1.bias, intermediate.dense.bias)
            self._copy_if_same_shape(decoder_layer.linear2.weight, output.dense.weight)
            self._copy_if_same_shape(decoder_layer.linear2.bias, output.dense.bias)
            self._copy_if_same_shape(decoder_layer.norm1.weight, attention.output.LayerNorm.weight)
            self._copy_if_same_shape(decoder_layer.norm1.bias, attention.output.LayerNorm.bias)
            self._copy_if_same_shape(decoder_layer.norm3.weight, output.LayerNorm.weight)
            self._copy_if_same_shape(decoder_layer.norm3.bias, output.LayerNorm.bias)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.encoder_call_count += 1
        with torch.no_grad():
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, target_length = decoder_input_ids.shape
        positions = torch.arange(target_length, device=decoder_input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, target_length)
        target = self.token_embedding(decoder_input_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(target_length, target_length, device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        target_padding_mask = decoder_input_ids.eq(self.config.pad_token_id)
        if self.config.use_cross_attention:
            memory = encoder_hidden_states
            memory_padding_mask = encoder_attention_mask.eq(0)
        else:
            memory = torch.zeros(
                batch_size,
                1,
                self.hidden_size,
                dtype=encoder_hidden_states.dtype,
                device=encoder_hidden_states.device,
            )
            memory_padding_mask = torch.zeros(
                batch_size,
                1,
                dtype=torch.bool,
                device=encoder_hidden_states.device,
            )
        decoded = self.decoder(
            tgt=target,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.lm_head(self.final_norm(decoded))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoder_hidden_states = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.decode(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )
        output = {"logits": logits}
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        repetition_penalty: float,
    ) -> torch.Tensor:
        if repetition_penalty == 1.0:
            return logits
        logits = logits.clone()
        for batch_index in range(generated.size(0)):
            seen_tokens = set(generated[batch_index].tolist())
            for token_id in seen_tokens:
                if logits[batch_index, token_id] < 0:
                    logits[batch_index, token_id] *= repetition_penalty
                else:
                    logits[batch_index, token_id] /= repetition_penalty
        return logits

    def _apply_no_repeat_ngram(
        self,
        logits: torch.Tensor,
        generated: torch.Tensor,
        no_repeat_ngram_size: int,
    ) -> torch.Tensor:
        if no_repeat_ngram_size <= 0 or generated.size(1) + 1 < no_repeat_ngram_size:
            return logits
        logits = logits.clone()
        prefix_length = no_repeat_ngram_size - 1
        for batch_index in range(generated.size(0)):
            tokens = generated[batch_index].tolist()
            if len(tokens) < prefix_length:
                continue
            current_prefix = tuple(tokens[-prefix_length:])
            banned_tokens: set[int] = set()
            for start in range(0, len(tokens) - no_repeat_ngram_size + 1):
                ngram = tuple(tokens[start : start + no_repeat_ngram_size])
                if ngram[:-1] == current_prefix:
                    banned_tokens.add(ngram[-1])
            if banned_tokens:
                logits[batch_index, list(banned_tokens)] = -torch.inf
        return logits

    def _top_k_top_p_filter(
        self,
        logits: torch.Tensor,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        filtered = logits.clone()
        if top_k > 0 and top_k < filtered.size(-1):
            threshold = torch.topk(filtered, top_k, dim=-1).values[:, -1].unsqueeze(-1)
            filtered = filtered.masked_fill(filtered < threshold, -torch.inf)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove = cumulative_probs > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
            filtered = torch.full_like(filtered, -torch.inf)
            filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        return filtered

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        num_beams: int = 1,
        length_penalty: float = 1.0,
        min_new_tokens: int = 0,
    ) -> torch.Tensor:
        self.eval()
        encoder_hidden_states = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        if num_beams > 1:
            if do_sample:
                raise ValueError("Beam search does not support sampling in this implementation.")
            if input_ids.size(0) != 1:
                raise ValueError("Beam search currently supports batch_size=1.")
            return self._generate_beam_search(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                min_new_tokens=min_new_tokens,
            )
        generated = torch.full(
            (input_ids.size(0), 1),
            self.config.bos_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )
        for _ in range(max_new_tokens):
            logits = self.decode(
                decoder_input_ids=generated,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
            )
            next_logits = logits[:, -1]
            next_logits = self._apply_repetition_penalty(
                next_logits,
                generated,
                repetition_penalty=repetition_penalty,
            )
            next_logits = self._apply_no_repeat_ngram(
                next_logits,
                generated,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            if generated.size(1) - 1 < min_new_tokens:
                next_logits[:, self.config.eos_token_id] = -torch.inf
            if do_sample:
                scaled_logits = next_logits / max(temperature, 1e-6)
                filtered_logits = self._top_k_top_p_filter(
                    scaled_logits,
                    top_k=top_k,
                    top_p=top_p,
                )
                probs = torch.softmax(filtered_logits, dim=-1)
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.eq(self.config.eos_token_id)):
                break
        return generated

    def _length_normalized_score(
        self,
        score: float,
        sequence_length: int,
        length_penalty: float,
    ) -> float:
        generated_length = max(1, sequence_length - 1)
        return score / (generated_length ** length_penalty)

    def _generate_beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        length_penalty: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        min_new_tokens: int,
    ) -> torch.Tensor:
        beams: list[tuple[torch.Tensor, float, bool]] = [
            (
                torch.tensor(
                    [[self.config.bos_token_id]],
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                0.0,
                False,
            )
        ]
        for _ in range(max_new_tokens):
            candidates: list[tuple[torch.Tensor, float, bool]] = []
            for sequence, score, finished in beams:
                if finished:
                    candidates.append((sequence, score, True))
                    continue
                logits = self.decode(
                    decoder_input_ids=sequence,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                )
                next_logits = logits[:, -1]
                next_logits = self._apply_repetition_penalty(
                    next_logits,
                    sequence,
                    repetition_penalty=repetition_penalty,
                )
                next_logits = self._apply_no_repeat_ngram(
                    next_logits,
                    sequence,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
                if sequence.size(1) - 1 < min_new_tokens:
                    next_logits[:, self.config.eos_token_id] = -torch.inf
                log_probs = torch.log_softmax(next_logits, dim=-1)
                top_scores, top_tokens = torch.topk(log_probs, k=num_beams, dim=-1)
                for token_score, token_id in zip(top_scores[0], top_tokens[0], strict=True):
                    next_token = token_id.view(1, 1)
                    next_sequence = torch.cat([sequence, next_token], dim=1)
                    next_score = score + float(token_score.item())
                    next_finished = int(token_id.item()) == self.config.eos_token_id
                    candidates.append((next_sequence, next_score, next_finished))
            candidates.sort(
                key=lambda item: self._length_normalized_score(
                    item[1],
                    item[0].size(1),
                    length_penalty=length_penalty,
                ),
                reverse=True,
            )
            beams = candidates[:num_beams]
            if all(finished for _, _, finished in beams):
                break
        best_sequence, _, _ = max(
            beams,
            key=lambda item: self._length_normalized_score(
                item[1],
                item[0].size(1),
                length_penalty=length_penalty,
            ),
        )
        return best_sequence
