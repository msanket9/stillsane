"""Probe generation from logs.

Real log files are messy: mixed shapes, truncated lines, and above all enormously
repetitive. The tests that matter are the ones about surviving the mess and about
collapsing the repetition, not about the happy path.
"""

from __future__ import annotations

import json

import yaml

from stillsane.config import Config
from stillsane.generate import (
    Candidate,
    cluster,
    collect_candidates,
    extract_from_record,
    probe_id,
    read_records,
    to_yaml,
)
from stillsane.signals import HashingEmbedder

EMBEDDER = HashingEmbedder()


# --- Reading the shapes people actually have -------------------------------


def test_reads_an_openai_request_body():
    got = extract_from_record(
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Extract the total."},
            ],
        }
    )
    assert got == ("Extract the total.", "Be terse.")


def test_uses_the_last_user_turn_in_a_conversation():
    """A multi-turn log entry: the final user message is the actual request."""
    got = extract_from_record(
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "Now extract the total please."},
            ]
        }
    )
    assert got[0] == "Now extract the total please."


def test_reads_the_multipart_content_form():
    got = extract_from_record(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "Do the thing."}]}]}
    )
    assert got == ("Do the thing.", None)


def test_reads_a_bare_prompt_field():
    assert extract_from_record({"prompt": "Summarise this."}) == ("Summarise this.", None)


def test_unwraps_a_logging_envelope():
    """Most people log `{"ts": ..., "request": {...}}` rather than the bare body."""
    got = extract_from_record(
        {"ts": "2026-07-31", "request": {"messages": [{"role": "user", "content": "Nested ask."}]}}
    )
    assert got == ("Nested ask.", None)


def test_declines_rather_than_guesses():
    assert extract_from_record({"unrelated": "log line"}) is None
    assert extract_from_record("a string") is None
    assert extract_from_record({"messages": []}) is None


# --- Surviving a messy file ------------------------------------------------


def test_a_truncated_line_does_not_lose_the_file(tmp_path):
    """Refusing a 10,000-line log because one line was cut mid-write would make
    this useless on exactly the files it exists for."""
    log = tmp_path / "requests.jsonl"
    log.write_text(
        json.dumps({"prompt": "First real prompt here."})
        + "\n"
        + '{"prompt": "truncated mid-wri'
        + "\n"
        + json.dumps({"prompt": "Second real prompt here."})
        + "\n"
    )
    got = collect_candidates(read_records(log))
    assert [c.prompt for c in got] == ["First real prompt here.", "Second real prompt here."]


def test_reads_a_json_array(tmp_path):
    log = tmp_path / "requests.json"
    log.write_text(json.dumps([{"prompt": "Alpha prompt text."}, {"prompt": "Beta prompt text."}]))
    assert len(collect_candidates(read_records(log))) == 2


def test_reads_a_directory_of_json_files(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    (d / "a.json").write_text(json.dumps({"prompt": "Alpha prompt text."}))
    (d / "b.json").write_text(json.dumps({"prompt": "Beta prompt text."}))
    (d / "c.json").write_text("not json at all")
    assert len(collect_candidates(read_records(d))) == 2


def test_short_fragments_are_skipped(tmp_path):
    log = tmp_path / "r.jsonl"
    log.write_text(json.dumps({"prompt": "hi"}) + "\n" + json.dumps({"prompt": "A real prompt."}))
    assert [c.prompt for c in collect_candidates(read_records(log))] == ["A real prompt."]


# --- Collapsing the repetition ---------------------------------------------


def test_exact_duplicates_are_counted_not_repeated(tmp_path):
    log = tmp_path / "r.jsonl"
    log.write_text("\n".join(json.dumps({"prompt": "Extract the invoice total."}) for _ in range(5)))
    got = collect_candidates(read_records(log))
    assert len(got) == 1 and got[0].count == 5


def test_most_frequent_first():
    log = [{"prompt": "Rare prompt about weather."}] + [
        {"prompt": "Common prompt about invoices."} for _ in range(4)
    ]
    got = collect_candidates(log)
    assert got[0].prompt.startswith("Common") and got[0].count == 4


def test_near_duplicates_are_merged():
    """The real problem with logs: a thousand requests are a handful of shapes."""
    candidates = [
        Candidate("Extract the total from invoice 4471.", count=9),
        Candidate("Extract the total from invoice 4472.", count=7),
        Candidate("Extract the total from invoice 4473.", count=5),
        Candidate("Write a haiku about the sea.", count=2),
    ]
    merged = cluster(candidates, EMBEDDER, merge_distance=0.3)
    assert len(merged) == 2
    assert merged[0].prompt.endswith("4471.")  # the most frequent survives
    assert merged[0].count == 21  # counts fold into the survivor


def test_genuinely_different_prompts_are_kept_apart():
    candidates = [
        Candidate("Extract the total from this invoice.", count=3),
        Candidate("Translate the following text into French.", count=2),
        Candidate("Write a haiku about the sea.", count=1),
    ]
    assert len(cluster(candidates, EMBEDDER, merge_distance=0.12)) == 3


def test_clustering_a_single_candidate_is_a_no_op():
    one = [Candidate("Only one prompt here.", count=1)]
    assert cluster(one, EMBEDDER) == one
    assert cluster([], EMBEDDER) == []


# --- Emitting config -------------------------------------------------------


def test_generated_yaml_parses_and_validates():
    """The whole feature is worthless if the file it writes is not loadable."""
    from stillsane.cli import TARGET_STANZA

    candidates = [
        Candidate('Extract "total" and due_date: reply as JSON.\nInvoice #4471.', count=3),
        Candidate("Translate the following text into French.", count=1),
    ]
    text = TARGET_STANZA + to_yaml(candidates)
    cfg = Config.model_validate(yaml.safe_load(text))
    assert len(cfg.probes) == 2
    assert cfg.probes[0].prompt.startswith('Extract "total"')


def test_awkward_prompt_characters_survive_the_round_trip():
    """Prompts are full of quotes, colons and newlines. A literal block handles
    all of them, which is why it is used instead of quoting."""
    nasty = 'Reply with: {"a": 1}\n  indented: yes\n- dashed: too\n"quoted"'
    parsed = yaml.safe_load(to_yaml([Candidate(nasty, count=1)]))
    assert parsed["probes"][0]["prompt"].strip() == nasty


def test_system_prompts_are_carried_through():
    parsed = yaml.safe_load(
        to_yaml([Candidate("Do the thing please.", system="You are terse.", count=1)])
    )
    assert parsed["probes"][0]["system"].strip() == "You are terse."


def test_checks_are_commented_out_not_guessed():
    """Emitting a check that does not hold would fail the user's first baseline
    and teach them the tool is wrong. They get suggestions, commented."""
    text = to_yaml([Candidate("Extract the total as JSON.", count=1)])
    parsed = yaml.safe_load(text)
    assert "checks" not in parsed["probes"][0]
    assert "#   - valid_json" in text


def test_frequency_is_recorded_as_a_comment():
    text = to_yaml([Candidate("Extract the total.", count=42)])
    assert "seen 42 times" in text


def test_limit_caps_the_output():
    many = [Candidate(f"Prompt number {i} about something.", count=i) for i in range(30)]
    assert len(yaml.safe_load(to_yaml(many, limit=5))["probes"]) == 5


# --- Ids -------------------------------------------------------------------


def test_ids_are_readable():
    assert probe_id("Extract the total from this invoice", set()) == "extract_total_invoice"


def test_ids_are_unique():
    taken: set[str] = set()
    a = probe_id("Extract the total", taken)
    b = probe_id("Extract the total", taken)
    assert a != b and b.endswith("_2")


def test_ids_survive_an_unhelpful_prompt():
    assert probe_id("the a of for", set()) == "probe"
    assert probe_id("!!! ???", set()) == "probe"
