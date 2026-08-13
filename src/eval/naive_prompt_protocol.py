"""Shared prompt boundaries for genuinely Naive generation.

Naive means the first-stage prompt contains only the dataset content and the
assistant boundary.  It does not add an expert persona, an answer cue, or
other first-stage coaching.  CoT still uses the benchmark's configured second
answer stage; NoCoT is a single direct generation.

CoT intentionally ends at the incomplete ``<think`` prefix so the model emits
the closing ``>`` and continues naturally.  NoCoT owns the complete empty
``<think></think>`` block in the prompt and starts generation after it.
"""

from __future__ import annotations

import re


NAIVE_COT_ASSISTANT_PREFIX = "<think"
NAIVE_NOCOT_ASSISTANT_PREFIX = "<think></think>"
_LEGACY_GENERATED_CLOSER_PREFIX = "<think></think"
_GENERATED_CLOSER_RE = re.compile(r"^\s*>[ \t]*\r?\n?")


def strip_generated_empty_think_closer(text: str) -> str:
    """Normalize historical outputs that generated the legacy closer byte."""

    return _GENERATED_CLOSER_RE.sub("", str(text or ""), count=1)


def is_naive_nocot_prompt(prompt: str) -> bool:
    assistant_turn = str(prompt or "").rsplit("Assistant:", 1)[-1].lstrip()
    return assistant_turn.startswith(NAIVE_NOCOT_ASSISTANT_PREFIX) or assistant_turn.startswith(
        _LEGACY_GENERATED_CLOSER_PREFIX
    )


__all__ = [
    "NAIVE_COT_ASSISTANT_PREFIX",
    "NAIVE_NOCOT_ASSISTANT_PREFIX",
    "is_naive_nocot_prompt",
    "strip_generated_empty_think_closer",
]
