from __future__ import annotations

import random

from src.data import SpanTargetConfig, make_span_target_pair


def test_make_span_target_pair_removes_target_from_source():
    pair = make_span_target_pair(
        "alpha beta gamma delta epsilon zeta eta theta",
        random.Random(7),
        SpanTargetConfig(
            sentinel_token="[unused1]",
            min_span_length=2,
            max_span_length=2,
            min_left_words=2,
            min_right_words=2,
        ),
    )

    assert pair is not None
    assert "[unused1]" in pair.source
    assert pair.target not in pair.source
    assert len(pair.target.split()) == 2
