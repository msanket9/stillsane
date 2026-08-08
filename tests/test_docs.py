"""The docs describe commands that exist.

This suite exists because the README went stale three times in one week, every
time as a side effect of a change somewhere else, and every time it was caught by
someone reading carefully rather than by anything automatic. A README is the first
thing a new user runs, and it ships as the PyPI long description, so a command that
does not exist is not a documentation bug: it is the install appearing broken.

Deliberately checks the CLI surface rather than command output. Reproducing the
worked example needs the real embedder, which means a model download, and the
default test run has to stay offline. Names and flags are the part that can be
verified for free, and they cover the failure that actually happened: the README
describing `stillsane bands` while the released build had no such command.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from stillsane.cli import build_parser

DOCS = [
    Path(__file__).resolve().parents[1] / "README.md",
    Path(__file__).resolve().parents[1] / "examples" / "invoice-extract" / "README.md",
]

#: `stillsane` appears in prose constantly ("stillsane learns the band", "stillsane
#: is not an eval framework"), so only code contexts count: fenced blocks and inline
#: spans. Matching prose would make this test a nuisance rather than a guard.
_FENCED = re.compile(r"```[a-z]*\n(.*?)```", re.S)
#: Spans wrap across lines in prose (`stillsane\nwatch`), so newlines are allowed
#: and the whitespace is normalised afterwards. Bounded in length because an
#: unbalanced backtick would otherwise swallow whole paragraphs.
_INLINE = re.compile(r"`([^`]{1,120})`", re.S)
_INVOCATION = re.compile(r"^\s*(?:\$\s*)?stillsane\s+(.*)$")


def parser_surface() -> tuple[dict[str, set[str]], set[str]]:
    """Subcommand names to their flags, plus the top-level flags."""
    parser = build_parser()
    subs: dict[str, set[str]] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                subs[name] = {opt for a in sub._actions for opt in a.option_strings}
    top = {opt for a in parser._actions for opt in a.option_strings}
    return subs, top


def documented_invocations() -> list[tuple[Path, str]]:
    """Every `stillsane ...` written in a code context across the docs."""
    found: list[tuple[Path, str]] = []
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        chunks = [line for block in _FENCED.findall(text) for line in block.splitlines()]
        chunks += [" ".join(span.split()) for span in _INLINE.findall(text)]
        for chunk in chunks:
            match = _INVOCATION.match(chunk)
            if match:
                found.append((path, match.group(1).strip()))
    return found


def test_docs_contain_invocations_to_check():
    """A guard on the guard: a regex that silently matches nothing proves nothing."""
    assert len(documented_invocations()) >= 10


@pytest.mark.parametrize("path,invocation", documented_invocations())
def test_documented_command_exists(path: Path, invocation: str):
    """Every documented subcommand is one the CLI actually has.

    The failure this catches: 0.0.5 shipped a README describing `stillsane bands`
    while the build had no such command, so following the docs produced an argparse
    error on the first step.
    """
    subs, top = parser_surface()
    word = invocation.split()[0] if invocation.split() else ""
    if word.startswith("-"):
        assert word in top, f"{path.name}: top-level flag {word!r} does not exist"
        return
    assert word in subs, (
        f"{path.name}: documents `stillsane {word}`, which is not a command. "
        f"Available: {', '.join(sorted(subs))}"
    )


@pytest.mark.parametrize("path,invocation", documented_invocations())
def test_documented_flags_exist(path: Path, invocation: str):
    """Flags shown for a command belong to that command.

    Catches a rename landing in the code and not the docs, which is the same class
    of error as a missing command but quieter: the command runs and then rejects the
    argument the README told you to pass.
    """
    subs, top = parser_surface()
    parts = invocation.split()
    if not parts or parts[0].startswith("-") or parts[0] not in subs:
        return  # covered by the command test above

    allowed = subs[parts[0]] | top
    for token in parts[1:]:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        # Placeholders like `--probe <id>` are documentation, not arguments.
        if "<" in flag or ">" in flag:
            continue
        assert flag in allowed, (
            f"{path.name}: `stillsane {parts[0]}` is shown with {flag!r}, "
            f"which it does not accept. Accepts: {', '.join(sorted(allowed))}"
        )


def test_every_command_is_documented_somewhere():
    """A command nobody can discover may as well not exist.

    The reverse direction of the checks above: it is just as easy to ship a feature
    and forget the README as to document one that was removed.
    """
    subs, _ = parser_surface()
    documented = {inv.split()[0] for _, inv in documented_invocations() if inv.split()}
    missing = sorted(set(subs) - documented)
    assert not missing, f"CLI commands with no mention in the docs: {', '.join(missing)}"
