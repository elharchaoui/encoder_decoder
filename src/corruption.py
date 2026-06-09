from __future__ import annotations

import random
import re
from dataclasses import dataclass


ARTICLES_AND_PREPOSITIONS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
}


@dataclass(frozen=True)
class CorruptionConfig:
    token_delete_prob: float = 0.08
    token_mask_prob: float = 0.08
    function_word_delete_prob: float = 0.35
    span_delete_prob: float = 0.0
    span_mask_prob: float = 0.0
    min_span_length: int = 2
    max_span_length: int = 5
    shuffle_window_prob: float = 0.4
    min_keep_ratio: float = 0.35
    remove_punctuation: bool = True


def corruption_config_from_dict(values: dict | None) -> CorruptionConfig:
    if not values:
        return CorruptionConfig()
    allowed = set(CorruptionConfig.__dataclass_fields__)
    filtered = {key: value for key, value in values.items() if key in allowed}
    return CorruptionConfig(**filtered)


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _apply_span_corruption(
    words: list[str],
    rng: random.Random,
    mask_token: str,
    config: CorruptionConfig,
) -> list[str]:
    if len(words) < config.min_span_length + 2:
        return words
    output = words[:]
    max_span = min(config.max_span_length, max(config.min_span_length, len(output) // 3))
    if max_span < config.min_span_length:
        return output
    if rng.random() < config.span_delete_prob:
        span_len = rng.randint(config.min_span_length, max_span)
        start = rng.randrange(0, len(output) - span_len + 1)
        del output[start : start + span_len]
    if output and rng.random() < config.span_mask_prob:
        span_len = rng.randint(config.min_span_length, min(max_span, len(output)))
        start = rng.randrange(0, len(output) - span_len + 1)
        output[start : start + span_len] = [mask_token]
    return output


def corrupt_text(
    text: str,
    rng: random.Random,
    mask_token: str = "[MASK]",
    config: CorruptionConfig | None = None,
) -> str:
    config = config or CorruptionConfig()
    words = normalize_text(text).split()
    if len(words) <= 3:
        return text

    words = _apply_span_corruption(words, rng, mask_token, config)
    min_keep = max(1, int(len(words) * config.min_keep_ratio))
    output: list[str] = []
    kept_count = 0
    for index, word in enumerate(words):
        remaining_after_this = len(words) - index - 1
        can_delete = kept_count + remaining_after_this >= min_keep
        p = rng.random()
        if p < config.token_delete_prob and can_delete:
            continue
        if p < config.token_delete_prob + config.token_mask_prob:
            output.append(mask_token)
            kept_count += 1
            continue
        if (
            word in ARTICLES_AND_PREPOSITIONS
            and can_delete
            and rng.random() < config.function_word_delete_prob
        ):
            continue
        output.append(word)
        kept_count += 1

    if len(output) > 4 and rng.random() < config.shuffle_window_prob:
        start = rng.randrange(0, len(output) - 2)
        window = output[start : start + 3]
        rng.shuffle(window)
        output[start : start + 3] = window

    corrupted = " ".join(output)
    if config.remove_punctuation:
        corrupted = re.sub(r"[,.!?;:]", "", corrupted)
    return corrupted if corrupted else words[0]
