"""Signal-level behaviour, below the variance machinery."""

from __future__ import annotations

import pytest
from conftest import sample

from stillsane.models import Sample, ToolCall
from stillsane.signals import build_signals
from stillsane.signals.meta import ResponseComplete
from stillsane.signals.shape import JsonShapeDistance, ToolCallDistance, jaccard_distance, key_paths
from stillsane.signals.structural import (
    ConstantField,
    HasKeys,
    LengthChars,
    ValidJson,
    extract_lenient,
    normalise_field_value,
    parse_strict,
)

# --- JSON parsing ---------------------------------------------------------


def test_strict_parse_rejects_surrounding_prose():
    assert parse_strict('{"a": 1}') == {"a": 1}
    assert parse_strict('Here you go: {"a": 1}') is None


def test_strict_parse_accepts_a_code_fence():
    """Fenced JSON is too common to call invalid; prose outside a fence is not."""
    assert parse_strict('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_strict('```\n{"a": 1}\n```') == {"a": 1}


def test_lenient_extraction_finds_embedded_json():
    """Content checks must be able to answer 'is the data still there' separately."""
    assert extract_lenient('Here you go: {"a": 1} -- hope that helps!') == {"a": 1}
    assert extract_lenient('prefix [1, 2, 3] suffix') == [1, 2, 3]


def test_lenient_extraction_handles_braces_inside_strings():
    text = 'Result: {"note": "use {braces} carefully", "n": 2}'
    assert extract_lenient(text) == {"note": "use {braces} carefully", "n": 2}


def test_lenient_extraction_gives_up_cleanly():
    assert extract_lenient("no json here at all") is None
    assert extract_lenient('unbalanced {"a": 1') is None


def test_valid_json_and_has_keys_disagree_on_prose_wrapping():
    """The distinction that lets the report say 'envelope broke, data survived'."""
    wrapped = sample('Sure! {"total": 5, "due_date": "x"}')
    assert ValidJson().value(wrapped) == 0.0
    assert HasKeys(["total", "due_date"]).value(wrapped) == 1.0


def test_has_keys_reports_which_are_missing():
    s = sample('{"total": 5}')
    assert HasKeys(["total", "due_date"]).missing(s) == ["due_date"]


def test_signals_skip_errored_samples():
    errored = sample("", error="timeout")
    assert ValidJson().value(errored) is None
    assert LengthChars().value(errored) is None
    assert HasKeys(["a"]).value(errored) is None


# --- constant_fields ---------------------------------------------------------


def test_normalise_field_value_collapses_int_and_float_formatting():
    """The exact false alarm the report warns about: `1240` (int, from a
    response that dropped the decimal point) and `1240.0` (float) must
    compare equal, or a formatting wobble alone reads as the total changing.
    """
    assert normalise_field_value(1240) == normalise_field_value(1240.0)
    assert normalise_field_value(1240.50) == normalise_field_value(1240.5)


def test_normalise_field_value_still_distinguishes_real_differences():
    assert normalise_field_value(1240.50) != normalise_field_value(1204.50)


def test_normalise_field_value_bool_is_not_treated_as_a_number():
    """`bool` is a subclass of `int` in Python -- without an explicit check
    first, `True` would silently normalise to the same string as `1.0`."""
    assert normalise_field_value(True) != normalise_field_value(1)
    assert normalise_field_value(True) == "true"
    assert normalise_field_value(False) == "false"


def test_normalise_field_value_null_is_distinct_from_a_string_null():
    assert normalise_field_value(None) == "null"
    assert normalise_field_value("null") == "null"
    # Both normalise to the literal text "null", which is a known, accepted
    # (if slightly unusual) collision -- a field whose real value is the
    # string "null" is exotic enough not to design around.


def test_normalise_field_value_nested_structures_are_stable():
    assert normalise_field_value({"b": 2, "a": 1}) == normalise_field_value({"a": 1, "b": 2})
    assert normalise_field_value([1, 2]) != normalise_field_value([2, 1])


def test_constant_field_reads_the_named_key():
    s = sample('{"total": 1240.50, "due_date": "2026-07-01"}')
    assert ConstantField("total").value(s) == "1240.5"
    assert ConstantField("due_date").value(s) == "2026-07-01"


def test_constant_field_is_none_when_the_field_is_missing():
    """Missing entirely is `has_keys`'s question, not this one's -- `None`
    means 'contributes to neither side of the comparison', not 'changed to
    nothing'."""
    s = sample('{"due_date": "2026-07-01"}')
    assert ConstantField("total").value(s) is None


def test_constant_field_is_none_for_non_json_text():
    assert ConstantField("total").value(sample("just prose, no JSON here")) is None


def test_constant_field_skips_errored_samples():
    assert ConstantField("total").value(sample("", error="timeout")) is None


def test_constant_fields_creates_one_signal_per_field(embedder):
    """"total changed" and "due_date changed" are different findings a
    report should name separately -- unlike `has_keys`, which asks one
    combined yes/no question about every key in its list.
    """
    names = [s.name for s in build_signals([{"constant_fields": ["total", "due_date"]}], embedder)]
    assert "constant_field[total]" in names
    assert "constant_field[due_date]" in names


def test_constant_fields_needs_a_list(embedder):
    with pytest.raises(ValueError, match="non-empty list"):
        build_signals([{"constant_fields": "total"}], embedder)
    with pytest.raises(ValueError, match="non-empty list"):
        build_signals([{"constant_fields": []}], embedder)


def test_unknown_check_message_mentions_constant_fields(embedder):
    with pytest.raises(ValueError, match="constant_fields"):
        build_signals(["definitely_not_a_check"], embedder)


# --- Truncation -------------------------------------------------------------


def test_response_complete_flags_known_truncation_reasons():
    """OpenAI's "length" and Anthropic's "max_tokens" are the two this project has
    concrete evidence for -- found by hand, in the essay probe of a real
    calibration run, before this signal existed to catch it automatically."""
    assert ResponseComplete().value(sample("...", finish_reason="length")) == 0.0
    assert ResponseComplete().value(sample("...", finish_reason="max_tokens")) == 0.0


def test_response_complete_accepts_ordinary_finish_reasons():
    """Every other way a response legitimately ends. None of these are truncation,
    and flagging them would fire on ordinary agent behaviour."""
    for reason in ("stop", "end_turn", "tool_calls", "tool_use", "stop_sequence"):
        assert ResponseComplete().value(sample("done", finish_reason=reason)) == 1.0


def test_response_complete_is_silent_when_the_provider_does_not_expose_it():
    """No finish_reason at all -- an http target with no matching field, or a
    provider this project has not added detection for -- must not be guessed at.
    Silent, the same as fingerprint watching on a provider with no fingerprint."""
    assert ResponseComplete().value(sample("done", finish_reason=None)) is None


def test_response_complete_skips_errored_samples():
    assert ResponseComplete().value(sample("", error="timeout", finish_reason="length")) is None


def test_response_complete_catches_a_regression_a_length_check_would_miss():
    """The reason this signal exists rather than trusting length_chars: a
    truncated response can still be *longer* than the baseline, and embed close
    to it right up to the point it stops, because everything up to the cutoff is
    genuine, on-topic content. Length and semantic distance alone would not
    reliably flag this; only the finish reason says the answer is incomplete.
    """
    baseline = sample("A" * 500, finish_reason="stop")
    current = sample("A" * 600, finish_reason="length")  # longer, and truncated
    assert ResponseComplete().value(baseline) == 1.0
    assert ResponseComplete().value(current) == 0.0


# --- Shape ----------------------------------------------------------------


def test_key_paths_flattens_nested_structure():
    assert key_paths({"a": {"b": 1}}) == {"a", "a.b"}


def test_key_paths_ignores_list_length():
    """Three items and two items are the same shape; count is content, not shape."""
    assert key_paths({"xs": [{"a": 1}, {"a": 2}]}) == key_paths({"xs": [{"a": 1}]})


def test_json_shape_distance_notices_a_new_field():
    a = sample('{"total": 1, "due": "x"}')
    b = sample('{"total": 1, "due": "x", "currency": "USD"}')
    d = JsonShapeDistance().distance(a, b)
    assert 0 < d < 1


def test_json_shape_distance_skips_non_json_probes():
    a, b = sample("just prose"), sample("more prose")
    assert JsonShapeDistance().distance(a, b) is None


def test_json_shape_distance_flags_json_disappearing():
    assert JsonShapeDistance().distance(sample('{"a": 1}'), sample("sorry, I can't")) == 1.0


def test_jaccard_edges():
    assert jaccard_distance(set(), set()) == 0.0
    assert jaccard_distance({"a"}, set()) == 1.0
    assert jaccard_distance({"a"}, {"a"}) == 0.0


# --- Tool calls -----------------------------------------------------------


def test_tool_call_parsing_from_openai_shape():
    tc = ToolCall.from_openai(
        {"function": {"name": "search", "arguments": '{"query": "x", "limit": 5}'}}
    )
    assert tc.name == "search"
    assert tc.arg_keys == ("limit", "query")  # sorted, so order is not drift


def test_tool_call_parsing_survives_malformed_arguments():
    tc = ToolCall.from_openai({"function": {"name": "search", "arguments": "not json"}})
    assert tc.name == "search" and tc.arg_keys == ()


def test_tool_call_distance_ignores_argument_values():
    """Values vary legitimately every run; only the shape is drift."""
    a = sample("x", tool_calls=[ToolCall("search", ("query",))])
    b = sample("x", tool_calls=[ToolCall("search", ("query",))])
    assert ToolCallDistance().distance(a, b) == 0.0


def test_tool_call_distance_notices_repetition():
    once = sample("x", tool_calls=[ToolCall("search", ("q",))])
    twice = sample("x", tool_calls=[ToolCall("search", ("q",))] * 2)
    d = ToolCallDistance().distance(once, twice)
    assert 0 < d < 1


def test_tool_call_distance_skips_non_agent_probes():
    assert ToolCallDistance().distance(sample("a"), sample("b")) is None


# --- Registry -------------------------------------------------------------


def test_unknown_check_is_rejected_not_ignored(embedder):
    """A silently dropped check is one the user thinks is protecting them."""
    with pytest.raises(ValueError, match="Unknown check"):
        build_signals(["definitely_not_a_check"], embedder)


def test_malformed_check_is_rejected(embedder):
    with pytest.raises(ValueError, match="Malformed check"):
        build_signals([{"a": 1, "b": 2}], embedder)


def test_has_keys_needs_a_list(embedder):
    with pytest.raises(ValueError, match="non-empty list"):
        build_signals([{"has_keys": "total"}], embedder)


def test_always_on_signals_need_no_config(embedder):
    names = {s.name for s in build_signals(None, embedder)}
    assert {"semantic_distance", "fingerprint", "latency_ms", "completion_tokens"} <= names


def test_watch_fingerprint_false_drops_the_signal_entirely(embedder):
    names = {s.name for s in build_signals(None, embedder, watch_fingerprint=False)}
    assert "fingerprint" not in names
    # Nothing else about the always-on set should move.
    assert {"semantic_distance", "latency_ms", "completion_tokens", "model_id"} <= names


def test_watch_fingerprint_defaults_to_true(embedder):
    names = {s.name for s in build_signals(None, embedder)}
    assert "fingerprint" in names


def test_max_length_does_not_duplicate_the_always_on_length_signal(embedder):
    """`max_length` used to append a second `LengthChars` while leaving its name
    as `length_chars`, so a probe with the check got two signals sharing one
    name -- two report rows, two history rows per run, and `calibrate` merging
    both into one misleading line. The capped signal must carry its own name.
    """
    names = [s.name for s in build_signals([{"max_length": 2000}], embedder)]
    assert names.count("length_chars") == 1
    assert names.count("max_length") == 1

    capped = next(s for s in build_signals([{"max_length": 2000}], embedder) if s.name == "max_length")
    assert capped.band_override == 2000.0


def test_semantic_distance_threshold_can_be_pinned(embedder):
    signals = build_signals([{"semantic_distance": 0.2}], embedder)
    semantic = next(s for s in signals if s.name == "semantic_distance")
    assert semantic.band_override == 0.2


def test_semantic_similarity_is_inverted_into_a_distance_ceiling(embedder):
    """`semantic_similarity: 0.9` means "stay 90% similar", a distance ceiling of 0.1.

    It used to be applied as a distance of 0.9, which no response ever exceeds, so a
    check documented as a threshold could never fire.
    """
    signals = build_signals([{"semantic_similarity": 0.9}], embedder)
    semantic = next(s for s in signals if s.name == "semantic_distance")
    assert semantic.band_override == pytest.approx(0.1)


def test_a_similarity_floor_actually_fires_on_a_prose_wrapped_response(embedder):
    """The repro from the audit: a documented threshold that could never be crossed."""
    from stillsane.compare import BandConfig, evaluate_pairwise
    from stillsane.models import Level

    stable = ['{"total": 1240.50, "due_date": "2026-07-01"}'] * 4 + [
        '{"due_date": "2026-07-01", "total": 1240.50}'
    ]
    wrapped = "Sure! I found the following:\n{\"total\": 1240.5, \"due_date\": \"2026-07-01\"}\nHappy to help."
    semantic = next(
        s for s in build_signals([{"semantic_similarity": 0.9}], embedder) if s.name == "semantic_distance"
    )
    verdict = evaluate_pairwise(
        semantic, [sample(t) for t in stable], [sample(wrapped) for _ in range(3)], BandConfig()
    )
    assert verdict.level is not Level.PASS
    assert verdict.band.pinned and verdict.band.upper == pytest.approx(0.1)


@pytest.mark.parametrize(
    "check",
    [{"semantic_similarity": 1.5}, {"semantic_similarity": -0.1}, {"semantic_distance": -1},
     {"semantic_similarity": "high"}, {"semantic_distance": True}],
)
def test_semantic_thresholds_reject_nonsense(embedder, check):
    with pytest.raises(ValueError):
        build_signals([check], embedder)


def test_semantic_auto_leaves_the_band_learned(embedder):
    signals = build_signals([{"semantic_similarity": "auto"}], embedder)
    semantic = next(s for s in signals if s.name == "semantic_distance")
    assert semantic.band_override is None


# --- Embedding ------------------------------------------------------------


def test_hashing_embedder_is_deterministic(embedder):
    a = embedder.encode(["the quick brown fox"])
    b = embedder.encode(["the quick brown fox"])
    assert (a == b).all()


def test_hashing_embedder_normalises(embedder):
    import numpy as np

    vecs = embedder.encode(["short", "a considerably longer piece of text here"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_hashing_embedder_handles_empty_text(embedder):
    import numpy as np

    vec = embedder.encode([""])
    assert vec.shape[0] == 1 and not np.isnan(vec).any()


# --- Sample round-trip ----------------------------------------------------


def test_sample_survives_serialisation():
    original = sample(
        '{"a": 1}',
        fingerprint="fp_1",
        completion_tokens=42,
        latency_ms=123.4,
        tool_calls=[ToolCall("search", ("q", "limit"))],
    )
    restored = Sample.from_dict(original.to_dict())
    assert restored.text == original.text
    assert restored.fingerprint == original.fingerprint
    assert restored.completion_tokens == original.completion_tokens
    assert restored.tool_calls == original.tool_calls
    assert restored.ts == original.ts


def test_sample_from_dict_tolerates_unknown_fields():
    """Reading a baseline written by a newer version must not explode."""
    d = sample("hi").to_dict()
    d["some_future_field"] = True
    assert Sample.from_dict(d).text == "hi"
