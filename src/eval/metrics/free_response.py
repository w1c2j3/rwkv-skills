"""Free-response evaluation using full completions and math_verify."""

from __future__ import annotations

import ast
import html as html_lib
import hashlib
import json
import re
import os
import signal
import time
import unicodedata
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import tqdm
import httpx
from openai import OpenAI

from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.datasets.data_struct.free_answer import FreeAnswerRecord
from src.eval.k_values import NumericK, filter_metrics_by_k
from src.eval.metrics.at_k import compute_avg_at_k, compute_pass_at_k
from src.eval.naive_prompt_protocol import (
    is_naive_nocot_prompt,
    strip_generated_empty_think_closer,
)
from src.eval.results.io import iter_jsonl
from src.eval.results.schema import make_eval_payload, strict_nonneg_int

USER_SENTINEL = "\nUser:"
LEGACY_GENERATION_STOP_SUFFIXES = (USER_SENTINEL,)
G1H_GENERATION_STOP_SUFFIXES = (
    USER_SENTINEL,
    "\nUser✿",
    "User✿",
    "\nBot✿",
    "Bot✿",
    "\nAssistant:",
    "Assistant:",
    "✿",
)
REPAIR_FINAL_CUE = "Therefore, the final answer is "
MISSING_STRATEGY_A_PREDICTION = "missing_strategy_a_prediction"
MISSING_RECOVERY_PREDICTION = "missing_recovery_prediction"
STRATEGY_A = "strategy_a"
STRATEGY_B = "strategy_b"
STRATEGY_C = "strategy_c"
STRATEGY_GROUPS = (STRATEGY_A, STRATEGY_B, STRATEGY_C)
STRATEGY_LABELS = {
    STRATEGY_A: "strategy_a",
    STRATEGY_B: "strategy_b",
    STRATEGY_C: "strategy_c",
}

_WHITESPACE_RE = re.compile(r"\s+")
_JUDGEMENT_LABEL_RE = re.compile(
    r"\bjudg(?:e)?ment[\"']?\s*:\s*[\"']?(yes|no)\b",
    re.IGNORECASE,
)
_TRAILING_JUDGEMENT_LABEL_RE = re.compile(
    r"(?:\bjudg(?:e)?ment\s*:\s*)?\b(yes|no)\b\s*[.!。]?\s*$",
    re.IGNORECASE,
)
_FINAL_INTEGER_RE = re.compile(
    r"(?:final\s+answer|answer\s+is|the\s+answer|therefore)[^\n\r]{0,240}?([+-]?\d[\d,]*)",
    re.IGNORECASE,
)
_SIMPLE_INTEGER_RE = re.compile(r"^[+-]?\d+(?:\.0+)?$")
_PREFERRED_ANSWER_KEYS = (
    "expected_judgement",
    "expected_answer",
    "reference_answer",
    "target",
    "final_answer",
)
_SERIALIZED_ANSWER_KEYS = (
    "answer",
    "final_answer",
    "output",
    "response",
    "completion",
    "content",
    "text",
    "choices",
    "message",
)
_ANSWER_WINDOW_MARKERS = (
    "\\boxed",
    "final answer",
    "answer is",
    "answer:",
    "the answer",
    "therefore",
)
_ANSWER_WINDOW_PREFIX_CHARS = 400
_ANSWER_WINDOW_SUFFIX_CHARS = 1800
_ANSWER_WINDOW_TAIL_CHARS = 2500
_DEFAULT_MATH_VERIFY_TIMEOUT_S = 2.0
_MAX_DETERMINISTIC_SYMBOL_BIJECTIONS = 24
_SYMBOL_BIJECTION_LIMIT_FAIL_REASON = "math_verify_symbol_bijection_limit"
_MATH_VERIFY_GRADER_LOCK = threading.RLock()
_MISSING_SYMBOL_ATTRIBUTE = object()
# The global audit uses this marker to avoid re-enabling math-verify's nested
# timer while temporarily extending the scorer-owned deadline.
_verify_with_configured_timeout = True
_LATEX_OPTION_MARKER_RE = re.compile(
    r"(?i)\\(?:text|mathrm|mathbf|textbf)\s*\{\s*([A-Z])\s*(?:\)|\.|:)?\s*\}"
)
_PAREN_OPTION_MARKER_RE = re.compile(r"(?i)\(\s*([A-Z])\s*\)")
_SQUARE_OPTION_MARKER_RE = re.compile(r"(?i)\[\s*([A-Z])\s*\]")
_CURLY_OPTION_MARKER_RE = re.compile(
    # Do not reinterpret the payload of ``\text{A}`` as a second marker.
    # Parenthesised/square markers intentionally have no left boundary because
    # real source records often concatenate them as ``alpha(B) beta``.
    r"(?i)(?<![\\\w])\{\s*([A-Z])\s*\}"
)
_DELIMITED_OPTION_MARKER_RE = re.compile(
    r"(?im)(?:^|(?<=[\s;]))(?:\\noindent\s*)?([A-Z])\s*(?:\)|\.|:)\s+"
)
_OPTION_SCHEMA_HINT_RE = re.compile(
    r"(?im)(?:[\(\[\{]\s*[A-Z]\s*[\)\]\}]|"
    r"(?:^|[\s;])(?:\\noindent\s*)?(?:option\s+)?[A-Z]\s*(?:\)|\.|:)\s+)",
)
_OPTION_LABEL_SEQUENCES = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "FGHJK")
_EXPLICIT_OPTION_LABEL_RE = re.compile(
    r"(?i:\b(?:final\s+answer|answer|correct\s+(?:answer|option|choice)|correct\s+choice)\b)"
    r"\s*(?::|=|\bis\b)?\s*(?:(?i:option|choice)\s*)?"
    r"(?:[*_`]+\s*)?[\(\[]?\s*((?i:[A-Z]))\s*[\)\]]?"
    r"(?:\s*[*_`]+)?(?=\s*(?:[.,;:!?]|$))"
)
_CONCLUSIVE_OPTION_LABEL_RE = re.compile(
    r"(?i:\b(?:corresponds?\s+to|maps?\s+to|equivalent\s+to)\s+)"
    r"(?:(?i:the)\s+)?(?:(?i:answer|response)\s+)?"
    r"(?:(?i:is)\s+)?(?:(?i:option|choice)\s+)((?i:[A-Z]))\b"
)
_FINAL_ANSWER_VALUE_RE = re.compile(
    r"(?i:\b(?:final\s+answer|answer|correct\s+(?:answer|option|choice))\b)"
    r"\s*(?::|=|\bis\b)?\s*([^\n\r]{1,500})"
)
_EXPLICIT_ANSWER_CUE_RE = re.compile(
    r"(?im:^[ \t]*(?:(?:[-+*>]\s+)|(?:#{1,6}\s*)|[*_`]+)*"
    r"final\s+answer"
    r"(?:\s*[:\-])?\s*[*_`]*\s*$)|"
    r"(?i:(?:\b(?:final\s+answer|the\s+answer|answer)\b|"
    r"\bI(?:'ll|\s+will|\s+shall)\s+answer\b|\bmy\s+answer\b|"
    r"\b(?:correction|corrected\s+answer|revised\s+answer)\b|"
    r"\b(?:the\s+)?correct\s+result\b))"
    r"\s*(?:(?::|=|,|;)\s*|(?:is|should\s+be|would\s+be|equals?)\b\s*|"
    r"(?=\\boxed\s*\{)|(?=(?:\\?\$)?\s*[+-]?\d))"
)
_ANSWER_COMMITMENT_CUE_RE = re.compile(
    r"(?i)^\s*(?:I(?:'ll|\s+will|\s+shall)\s+answer|my\s+answer)\b"
)
_ANSWER_REPLACEMENT_CUE_RE = re.compile(
    r"(?i)^\s*(?:correction|corrected\s+answer|revised\s+answer|"
    r"(?:the\s+)?correct\s+result|therefore|actually|instead)\b"
)
_BARE_REPLACEMENT_CUE_RE = re.compile(
    r"(?ix)\b(?:therefore|actually|however|rather|instead|in\s+fact|"
    r"on\s+(?:second\s+thought|reflection|reconsideration))\b"
)
_REPLACEMENT_ANSWER_PREFIX_RE = re.compile(
    r"(?ix)^\s*(?:"
    r"(?:the\s+)?(?:correct\s+)?(?:answer|value|result|option|choice)|"
    r"it|this"
    r")\s*(?:is|should\s+be|would\s+be|=|:)\s*"
)
_REPLACEMENT_CORRECT_SUFFIX_RE = re.compile(
    r"(?ix)^\s*(?P<payload>.+?)\s+(?:is|was)\s+"
    r"(?:the\s+)?(?:correct|right)(?:\s+(?:answer|value|result|option|choice))?"
    r"\s*[.!\u3002\uff01]?\s*$"
)
_NON_ANSWER_PAYLOAD_RE = re.compile(
    r"(?ix)^\s*(?:"
    r"(?:is\s+)?(?:correct|incorrect|wrong|right|valid|invalid|verified|confirmed)"
    r"|(?:is\s+)?(?:likely|probably|possibly|apparently)\s+"
    r"(?:correct|incorrect|wrong|right|unique|valid|invalid)"
    r"|(?:is\s+)?(?:unknown|unclear|undetermined|ambiguous|unique)"
    r")\s*[.!。！？]?\s*$"
)
_CONCRETE_MATH_ANSWER_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:"
    r"[+-]?\s*(?:\d+(?:[.,]\d+)*|\.\d+)"
    r"|[+-]?\s*\\(?:boxed|frac|dfrac|tfrac|sqrt|begin)\b"
    r"|[\(\[\{]\s*[+-]?\s*(?:\d|\.\d|\\(?:frac|dfrac|tfrac|sqrt)\b)"
    r"|(?:option|choice)\s+[A-Z]\b"
    r"|[A-Z](?=\s*(?:[.)\]}]|$)))"
)
_CONCRETE_MATH_ANSWER_ANYWHERE_RE = re.compile(
    r"(?ix)(?:"
    r"[+-]?\s*(?:\d+(?:[.,]\d+)*|\.\d+)"
    r"|\\(?:boxed|frac|dfrac|tfrac|sqrt|begin)\b"
    r"|(?:option|choice)\s+[A-Z]\b"
    r")"
)
_NON_FINAL_ANSWER_CLAUSE_RE = re.compile(
    r"(?i)\b(?:in\s+the\s+form|depending\s+on|we\s+need\s+to|"
    r"need\s+to\s+(?:find|determine)|might\s+be|could\s+be|"
    r"not\s+(?:sure|certain)|for\s+subproblem|assuming\s+that|"
    r"should\s+be\s+(?:put\s+)?(?:inside|in)|"
    r"must\s+be\s+(?:put\s+)?(?:inside|in))\b"
)
_EMPTY_OR_PLACEHOLDER_BOX_RE = re.compile(
    r"(?ix)\\boxed\s*\{\s*(?:"
    r"[?._\u2026]+|\\(?:dots|cdots|ldots)|"
    r"(?:answer|response|result|value)(?:\s+here)?"
    r")?\s*\}"
)
_CONDITIONAL_ANSWER_PREFIX_RE = re.compile(
    r"(?i)\b(?:if|whether|assuming|suppose|supposing)\s+(?:that\s+)?(?:the\s+)?$"
)
_CONDITIONAL_ANSWER_SUFFIX_RE = re.compile(
    r"(?i)\b(?:if|unless|assuming(?:\s+that)?|provided(?:\s+that)?)\b"
)
_ANSWER_PRESENTATION_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:presented|reported|expressed|rounded|given|stated)\s+as\b"
)
_CONCLUSIVE_RESULT_VERB_RE = re.compile(
    r"(?i)\b(?:gives?|yields?|equals?|evaluates?\s+to|comes?\s+to)\s+"
)
_CONCLUSIVE_SCALAR_RHS_RE = re.compile(
    r"(?ix)^\s*"
    r"(?:[$`*_({\[]\s*)*"
    r"(?:"
    r"[+-]?(?:\d+(?:[.,]\d+)*|\.\d+)"
    r"|\\(?:frac|dfrac|tfrac|sqrt)\s*\{"
    r")"
)
_RETRACTED_ANSWER_PREFIX_RE = re.compile(
    r"(?is)^\s*(?:but\s+)?(?:(?:wait|hold\s+on).{0,120}?\b"
    r"(?:wrong|incorrect|conflict(?:ing)?|contradict(?:ion|ory)?|mistake)\b|"
    r"(?:wait|hold\s+on)[,;:]?\s*"
    r"(?:no\b|that(?:'s|\s+is)\s+(?:wrong|incorrect|not\s+right)|"
    r"I\s+(?:made|have\s+made)\s+(?:a\s+)?mistake)|"
    r"no\b|actually\s+no\b|"
    r"that(?:'s|\s+is)\s+(?:wrong|incorrect|not\s+right)|"
    r"this\s+is\s+(?:wrong|incorrect)|I\s+(?:made|have\s+made)\s+"
    r"(?:a\s+)?mistake)"
)
_EXPLICIT_CORRECTION_MARKER_RE = re.compile(
    r"(?is)(?:^|[.!?\n]\s*)(?:correction|corrected\s+answer|"
    r"revised\s+answer)\s*[:=]"
)
_RETRACTION_CUE_START_RE = re.compile(
    r"(?is)\b(?:but\s+)?(?:wait|hold\s+on)\b"
)
_CURRENT_ANSWER_RETRACTION_RE = re.compile(
    r"(?is)\b(?:"
    r"(?:(?:that|this|my)\s+answer|the\s+(?:previous|earlier|above)\s+answer)"
    r"\s+(?:was|is|seems?|looks?)\s+(?:wrong|incorrect|not\s+right)"
    r"|I\s+(?:was|am)\s+(?:wrong|incorrect)"
    r"|I\s+(?:made|have\s+made)\s+(?:a\s+)?mistake"
    r")\b"
)
_INTERVENING_RETRACTION_ANTECEDENT_RE = re.compile(
    r"(?is)\b(?:suppose|hypothetically|imagine|a\s+student|"
    r"(?:another|the\s+other)\s+"
    r"(?:answer|route|calculation|solution)|someone\s+(?:says?|claims?)|"
    r"quoted?|quotation)\b"
)
_CONTEXTUAL_EVIDENCE_SOURCE_RE = re.compile(
    r"(?is)\b(?:"
    r"(?:a|the)\s+source\s+(?:says?|claims?|reports?)|"
    r"according\s+to\b|"
    r"(?:the\s+)?(?:textbook|article|prompt|passage|author)\s+"
    r"(?:says?|claims?|reports?|states?)|"
    r"under\s+(?:that|this|the)\s+assumption"
    r")\b"
)
_PERSONAL_COMPUTATION_SOURCE_RE = re.compile(
    r"(?is)\baccording\s+to\s+(?:my|our)\s+"
    r"(?:calculation|computation|analysis|derivation|work|result)\b"
)
_ADOPTED_COMPUTATION_SOURCE_RE = re.compile(
    r"(?is)\baccording\s+to\s+(?:(?:my|our|the|this|that)\s+)?"
    r"(?:calculation|computation|analysis|derivation|work|result)\b"
)
_SOURCE_ADOPTION_STANCE_RE = re.compile(
    r"(?is)\b(?:(?:I|we)\s+(?:agree|confirm|accept|adopt|endorse)\b|"
    r"(?:my|our|the)\s+(?:calculation|computation|analysis|result)\s+agrees?\b|"
    r"(?:which|that|this|it)\s+agrees?\s+with\s+(?:my|our)\s+"
    r"(?:calculation|computation|analysis|result)\b|"
    r"(?:the|that|this)\s+assumption\s+"
    r"(?:holds?|is\s+(?:valid|true|satisfied))\b)"
)
_SOURCE_REJECTION_STANCE_RE = re.compile(
    r"(?is)\b(?:(?:I|we)\s+(?:reject|disagree|dispute|refute|deny)\b|"
    r"(?:the|that|this)\s+assumption\s+"
    r"(?:fails?|does\s+not\s+hold|is\s+(?:false|invalid))\b|"
    r"(?:(?:the|that|this|my|our)\s+)?"
    r"(?:calculation|computation|analysis|derivation|result)\s+"
    r"(?:fails?|does\s+not\s+hold|is\s+(?:wrong|false|incorrect|invalid))\b|"
    r"(?:which|that|this|it)\s+"
    r"(?:fails?|does\s+not\s+hold|is\s+(?:wrong|false|incorrect|invalid))\b|"
    r"(?:that|this|it)\s+(?:is|was)\s+(?:wrong|false|incorrect)\b)"
)
_SOURCE_NEGATED_ADOPTION_STANCE_RE = re.compile(
    r"(?is)\b(?:"
    r"(?:(?:I|we)\s+)?(?:do\s+not|don't|cannot|can't|never)\s+"
    r"(?:agree|confirm|accept|adopt|endorse)\b|"
    r"(?:(?:I|we)\s+)?(?:accept|adopt|endorse)\s+neither\b|"
    r"(?:(?:I|we)\s+)?agree\s+with\s+neither\b|"
    r"(?:(?:I|we)\s+)?confirm\s+only\b[^.!?\n]{0,160}\b"
    r"(?:false|wrong|incorrect|invalid|rejected?)\b"
    r")"
)
_EXPLICIT_RETRACTION_EVENT_RE = re.compile(
    r"(?is)\b(?:"
    r"I\s+(?:retract|withdraw|reject)\s+(?:it|that|this|my\s+answer|"
    r"the\s+(?:previous|earlier|above)\s+answer)|"
    r"(?:retract|withdraw|disregard|ignore)\s+(?:it|that|this|"
    r"(?:my|the\s+(?:previous|earlier|above))\s+answer)|"
    r"scratch\s+that|that\s+was\s+(?:wrong|incorrect|not\s+right)"
    r")\b"
)
_QUESTIONED_ANSWER_EVENT_RE = re.compile(
    r"(?is)\b(?:is|are|could|would|might|can|should)\s+"
    r"(?:the\s+)?(?:answer|option|choice)\s+(?:be\s+)?"
    r"(?:[A-Z]|[+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*[?\uff1f]"
)
_NEGATED_EVENT_SCOPE_RE = re.compile(
    r"(?is)(?:"
    r"\b(?:do|does|did|will|would|should|can|could|must|may|might)\s+not"
    r"(?:\s+[\w'-]+){0,5}\s*|"
    r"\b(?:don't|doesn't|didn't|won't|wouldn't|shouldn't|can't|cannot|"
    r"couldn't|mustn't)\b(?:\s+[\w'-]+){0,5}\s*|"
    r"\bnever\b(?:\s+[\w'-]+){0,5}\s*|"
    r"\bno\s+need\s+to\b(?:\s+[\w'-]+){0,5}\s*|"
    r"\b(?:refuse|refuses|refused|decline|declines|declined)\s+to\b"
    r"(?:\s+[\w'-]+){0,5}\s*|"
    r"\b(?:it|that|this)\s+is\s+(?:simply\s+)?false\s+that\s*|"
    r"\b(?:it|that|this)\s+is\s+not\s+true\s+that\s*"
    r")$"
)
_BARE_TERMINAL_ANSWER_RE = re.compile(
    r"(?ix)^(?:"
    r"[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
    r"(?:\s*/\s*[+-]?(?:\d+(?:\.\d+)?|\.\d+))?(?:\s*%|\s*[A-Za-z]+)?|"
    r"[A-Z]|"
    r"\\(?:boxed|frac|dfrac|tfrac|sqrt)\s*\{.+\}"
    r")$"
)
_INCIDENTAL_TERMINAL_BOX_CONTEXT_RE = re.compile(
    r"(?is)\b(?:intermediate|temporary|temporarily|illustrative|"
    r"for\s+example|substitut(?:e|es|ed|ing|ion)|alternative|"
    r"compar(?:e|es|ed|ing|ison))\b"
)
_POSTPOSED_REEVALUATION_EVENT_RE = re.compile(
    r"(?is)\b(?:recalculat\w*|recomput\w*|recheck\w*|verif\w*|"
    r"revis\w*|correct(?:ed|ion)?|review(?:ed|ing)?|"
    r"on\s+(?:second\s+thought|reflection|reconsideration))\b"
)
_POSTPOSED_CONCLUSIVE_MARKER_RE = re.compile(
    r"(?is)\b(?:therefore|thus|hence|finally|"
    r"final\s+(?:answer|result|value))\b"
)
_POSTPOSED_CHANGE_EVENT_RE = re.compile(
    r"(?is)\b(?:"
    r"(?:changes?|updates?|replaces?)\s+(?:the\s+)?"
    r"(?:answer|result|value|total)\s+(?:to|with)|"
    r"(?:answer|result|value|total)\s+(?:changes?|updates?)\s+to"
    r")\b"
)
_POSTPOSED_RESULT_PREDICATE_RE = re.compile(
    r"(?is)\b(?:answer|result|value|total)\s+"
    r"(?:is|equals?|becomes?|changes?\s+to|updates?\s+to)\b"
)
_PRESENTATION_ONLY_BRIDGE_RE = re.compile(
    r"(?is)^(?:\s|[*_`$]|\\(?:\[|\]|\(|\))|</?think>)*$"
)
_TERMINAL_CONCLUSION_CLOSER_RE = re.compile(
    r"(?is)^\s*(?:[,.;:]\s*)?(?:(?:which|thereby)\s+)?"
    r"(?:complet(?:e|es|ed|ing)|conclud(?:e|es|ed|ing)|"
    r"finish(?:es|ed|ing)?|clos(?:e|es|ed|ing))\s+"
    r"(?:the\s+)?(?:argument|solution|calculation|derivation|conclusion|proof)"
    r"\s*[.!\u3002\uff01]?\s*$"
)
_TRAILING_INCOMPLETE_OPERATOR_RE = re.compile(
    r"(?:[=+*/^_,:-]|(?<![eE])[+-]|\\(?:frac|sqrt|cdot|times|div|left|right|begin))\s*$"
)
_TRAILING_CONTINUATION_RE = re.compile(
    r"(?i)\b(?:and|or|but|because|therefore|thus|hence|then|wait|"
    r"is|are|equals?|be|to|of|with|by|from|as|actually|"
    r"compute|calculate|determine|evaluate|interpret)\s*$"
)
_LATEX_SPACING_COMMAND_RE = re.compile(
    r"\\(?:[,!;:]|quad|qquad|enspace|thinspace|medspace|thickspace|"
    r"negthinspace|hspace\s*\{[^{}]*\})"
)
_LATEX_SCRIPT_GROUP_RE = re.compile(r"[_^]\s*\{([^{}]*)\}")
_EXPLICIT_SCALAR_PREFIX_RE = re.compile(
    r"(?ix)^\s*(?:[*_`]+\s*)?"
    r"(?P<value>"
    r"(?:(?:\\\$|\$)\s*)?"
    r"[+-]?\s*"
    r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:\s*/\s*[+-]?\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?|\.\d+))?"
    r"(?:\s*(?:\\?%|Â°|degrees?|radians?|"
    r"mm|cm|km|m|inches?|feet|ft|yards?|miles?|"
    r"mg|kg|g|ml|liters?|litres?|"
    r"seconds?|minutes?|hours?|days?|years?|"
    r"dollars?|usd|cents?|cups?))?"
    r")"
    r"(?:\s*[*_`]+)?(?P<suffix>.*)$"
)
_EXPLANATORY_SCALAR_SUFFIX_RE = re.compile(
    r"(?is)^\s*,\s*(?:even\s+though|although|despite|because|since|"
    r"while|whereas|which\b|notwithstanding)\b"
)
_ANSWER_CORRECTION_PREFIX_RE = re.compile(
    r"(?is)(?:\bactually\b|\bcorrection\b|\brevised\b|\binstead\b|"
    r"\btherefore\b|\bhowever\b|\brather\b|\bin\s+fact\b|"
    r"\bon\s+(?:second\s+thought|reflection|reconsideration)\b|"
    r"\bwait\b[^\n]{0,40}\b(?:right|correct)\b)[^\n]{0,48}$"
)
_FINAL_ANSWER_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:[-+*>]\s+)|(?:#{1,6}\s*)|[*_`]+)*"
    r"final\s+answer"
    r"(?:\s*[:\-])?\s*[*_`]*\s*$"
)

DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE = (
    "You are a rigorous AI judge. Your task is to evaluate whether a student's "
    "answer is mathematically equivalent to the reference answer, based on "
    "the provided question and reference answer. Accept different wording or formatting "
    "only when the mathematical value is unchanged and all required components are present.\\n\\n"
    "Input:\\nQuestion: <Q>\\nReference Answer: <REF>\\n"
    "Student's Answer: <A>\\n\\nOutput Format:\\nStrictly adhere to the output format: Only output 'True' or 'False'."
)

# This version identifies the complete, persisted request/response contract for
# external free-response judging.  Bump it whenever a field below changes
# meaning or whenever the request/response parsing semantics change.
LLM_JUDGE_PROTOCOL_VERSION = "rwkv.free_response.llm_judge.v1"
LLM_JUDGE_RESPONSE_CONTRACT = "trimmed_exact_literal_true_false.v1"


def llm_judge_prompt_sha256(text: str) -> str:
    """Return the stable UTF-8 digest used to identify a Judge prompt template."""

    if not isinstance(text, str):
        raise TypeError("Judge prompt template must be a string")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\\ ", " ").replace("\u00a0", " ")
    normalized = _WHITESPACE_RE.sub(" ", normalized.strip())
    return normalized


def _parse_serialized_answer_envelope(value: str) -> object | None:
    """Parse a whole-output JSON/Python-string envelope, never prose quotes."""

    text = value.strip()
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        if len(text) >= 2 and text[0] == text[-1] == "'":
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return None
    return None


def _serialized_answer_identity(value: str) -> str:
    """Return a presentation-insensitive identity for envelope agreement."""

    text = _normalize_text(value).strip("`*_ ")
    text = re.sub(
        r"(?is)^(?:final\s+answer|the\s+answer|answer)\s*(?::|=|\bis\b)\s*",
        "",
        text,
    ).strip()
    boxed = re.fullmatch(r"\\boxed\s*\{\s*(.*?)\s*\}\s*[.!]?", text)
    if boxed is not None:
        text = boxed.group(1)
    return text.strip("`*_ $.!\u3002\uff01 ").casefold()


def _serialized_answer_leaves(
    value: object,
    *,
    depth: int = 0,
) -> tuple[list[str], bool]:
    """Collect answer-bearing leaves and flag disagreement fail-closed."""

    if depth >= 5:
        return [], False
    if isinstance(value, str):
        parsed = _parse_serialized_answer_envelope(value)
        if parsed is not None and parsed != value:
            nested, conflict = _serialized_answer_leaves(parsed, depth=depth + 1)
            if nested or conflict:
                return nested, conflict
        return [value.strip()], False
    if isinstance(value, (int, float, bool)):
        return [str(value)], False
    if isinstance(value, (list, tuple)):
        leaves: list[str] = []
        for item in value:
            nested, conflict = _serialized_answer_leaves(
                item,
                depth=depth + 1,
            )
            if conflict:
                return [], True
            leaves.extend(answer for answer in nested if answer)
        if len({_serialized_answer_identity(item) for item in leaves}) > 1:
            return [], True
        return leaves[:1], False
    if not isinstance(value, dict):
        return [], False

    leaves: list[str] = []
    for key in _SERIALIZED_ANSWER_KEYS:
        if key not in value:
            continue
        nested, conflict = _serialized_answer_leaves(
            value[key],
            depth=depth + 1,
        )
        if conflict:
            return [], True
        leaves.extend(item for item in nested if item)
    if len({_serialized_answer_identity(item) for item in leaves}) > 1:
        return [], True
    return leaves[:1], False


def _unwrap_serialized_answer_text_with_status(value: str) -> tuple[str, bool]:
    """Return unwrapped text plus a conflicting-answer-fields flag."""

    parsed = _parse_serialized_answer_envelope(value)
    if parsed is None:
        return value, False
    leaves, conflict = _serialized_answer_leaves(parsed)
    if conflict:
        return value, True
    if leaves:
        return leaves[0].strip(), False
    return value, False


def _unwrap_serialized_answer_text(value: str) -> str:
    """Unwrap a consistent whole-output answer envelope."""

    text, _conflict = _unwrap_serialized_answer_text_with_status(value)
    return text


def _is_exact_match(prediction: str, reference: str) -> bool:
    return bool(reference) and _normalize_text(prediction) == _normalize_text(reference)


def _extract_judgement_label(value: str) -> str | None:
    normalized = _normalize_text(value)
    if not normalized:
        return None
    # A two-stage judgement lane commonly emits a bare final ``Yes``/``No``
    # after a prompt which itself contains both allowed labels.  The final
    # generated token is the authoritative answer, so inspect the tail before
    # considering labelled occurrences elsewhere in the text.  This also
    # prevents the output-format instruction in the prompt from leaking a
    # constant ``No`` prediction into the evaluator.
    trailing = _TRAILING_JUDGEMENT_LABEL_RE.search(normalized[-200:])
    if trailing:
        return f"Judgement: {trailing.group(1).capitalize()}"
    matches = _JUDGEMENT_LABEL_RE.findall(normalized)
    if matches:
        return f"Judgement: {matches[-1].capitalize()}"
    return None


def _is_judgement_reference(reference: str) -> bool:
    return _extract_judgement_label(reference) is not None


def _reference_option_label(reference: str) -> str | None:
    """Return a bare MCQ label without treating arbitrary one-letter maths as MCQ.

    The caller must additionally prove that the question contains a structured
    option set containing this label.  Keeping those two checks separate is
    what prevents a free-response reference such as ``x`` from entering the
    multiple-choice path.
    """

    text = unicodedata.normalize("NFKC", _normalize_text(reference))
    text = text.strip().strip("$").strip()
    for command in ("\\boxed", "\\text", "\\mathrm", "\\mathbf"):
        match = re.fullmatch(rf"{re.escape(command)}\s*\{{\s*([^{{}}]+)\s*\}}", text)
        if match:
            text = match.group(1).strip()
    match = re.fullmatch(
        r"(?:(?i:option|choice)\s*)?[\(\[]?\s*((?i:[A-Z]))\s*[\)\]]?[.!]?",
        text,
    )
    return match.group(1).upper() if match else None


def _question_option_markers(question: str) -> list[tuple[str, int, int]]:
    markers = {
        (match.group(1).upper(), match.start(), match.end())
        for pattern in (
            _LATEX_OPTION_MARKER_RE,
            _PAREN_OPTION_MARKER_RE,
            _SQUARE_OPTION_MARKER_RE,
            _CURLY_OPTION_MARKER_RE,
            _DELIMITED_OPTION_MARKER_RE,
        )
        for match in pattern.finditer(question)
    }
    return sorted(markers, key=lambda marker: (marker[1], marker[2]))


def _option_markers_are_ambiguous(
    markers: list[tuple[str, int, int]],
) -> bool:
    marker_counts: dict[str, int] = {}
    for label, _start, _end in markers:
        marker_counts[label] = marker_counts.get(label, 0) + 1
    if any(count > 1 for count in marker_counts.values()):
        return True
    for index, marker in enumerate(markers):
        for other in markers[index + 1 :]:
            if other[1] >= marker[2]:
                break
            if marker != other:
                return True
    return False


def _ordered_option_marker_chain(
    markers: list[tuple[str, int, int]],
    *,
    required_label: str | None,
) -> list[tuple[str, int, int]]:
    """Select the strongest ordered option sequence from noisy question text.

    Some source records concatenate option markers directly after the previous
    option text, while others include unrelated markers such as ``(I)`` before
    the actual ``(A)``-``(D)`` block.  Build candidate ordered chains and rank
    them by whether they contain the reference label, then by length.  The
    second label sequence covers the common F/G/H/J/K convention that skips I.
    """

    best: list[tuple[str, int, int]] = []
    best_rank = (False, 0, False, 0)
    for sequence in _OPTION_LABEL_SEQUENCES:
        label_positions = {label: index for index, label in enumerate(sequence)}
        for start_index, start in enumerate(markers):
            sequence_index = label_positions.get(start[0])
            if sequence_index != 0:
                continue
            chain = [start]
            cursor = start_index + 1
            for expected_label in sequence[sequence_index + 1 :]:
                match_index = next(
                    (
                        index
                        for index in range(cursor, len(markers))
                        if markers[index][0] == expected_label
                    ),
                    None,
                )
                if match_index is None:
                    break
                chain.append(markers[match_index])
                cursor = match_index + 1
            labels = {marker[0] for marker in chain}
            rank = (
                required_label is None or required_label in labels,
                len(chain),
                chain[0][0] == "A",
                -chain[0][1],
            )
            if rank > best_rank:
                best = chain
                best_rank = rank
    if len(best) < 2:
        return []
    if required_label is not None and required_label not in {item[0] for item in best}:
        return []
    return best


def _clean_question_option_text(value: str) -> str:
    text = value.strip()
    environment_end = re.search(r"\\end\s*\{[^{}]+\}", text)
    if environment_end:
        text = text[: environment_end.start()].rstrip()
    text = re.sub(r"^(?:\\\s*)?&\s*", "", text)
    text = re.sub(r"(?:&\s*)?(?:\\\\)?\s*$", "", text)
    return text.strip()


def _parse_question_options(
    question: str,
    *,
    required_label: str | None = None,
) -> dict[str, str]:
    """Parse a conservative, ordered MCQ option block from a question.

    Parenthesised markers support same-line SAT/Gaokao-style records, while
    line markers support the common ``A. text`` JSONL form.  Ambiguous or
    repeated markers fail closed and leave the existing free-response scorer
    untouched.
    """

    raw_markers = _question_option_markers(question)
    markers = _ordered_option_marker_chain(
        raw_markers,
        required_label=required_label,
    )
    if not markers:
        return {}
    # Only markers participating in the selected option alphabet can make the
    # schema ambiguous.  Repeated unrelated Roman-numeral facts such as two
    # ``(I)`` markers must not invalidate an otherwise clean A-D option block.
    selected_labels = {marker[0] for marker in markers}
    relevant_markers = [
        marker for marker in raw_markers if marker[0] in selected_labels
    ]
    if _option_markers_are_ambiguous(relevant_markers):
        return {}
    options: dict[str, str] = {}
    for index, marker in enumerate(markers):
        end = markers[index + 1][1] if index + 1 < len(markers) else len(question)
        value = _clean_question_option_text(question[marker[2] : end])
        if not value:
            return {}
        options[marker[0]] = value
    return options


def _position_is_quoted(text: str, end: int) -> bool:
    """Return whether ``end`` is inside an ASCII or Unicode quotation."""

    context = text[:end]
    single_quotes = re.findall(r"(?<!\w)'|'(?!\w)", context)
    return bool(
        len(single_quotes) % 2
        or context.count('"') % 2
        or context.count("\u2018") > context.count("\u2019")
        or context.count("\u201c") > context.count("\u201d")
    )


def _source_evidence_stance(
    text: str,
    *,
    start: int,
    end: int,
) -> str | None:
    """Return ``adopted``/``rejected``/``reported`` for sourced evidence.

    Source attribution is not itself rejection.  The model may explicitly
    adopt it (including its own calculation or a holding assumption), reject
    it, or merely report it.  A rejection event before a later answer event
    also ends the source's scope, preventing the later replacement from being
    mislabeled as part of the quotation.
    """

    search_start = max(0, start - 500)
    source_matches = list(
        _CONTEXTUAL_EVIDENCE_SOURCE_RE.finditer(text, search_start, start + 1)
    )
    if not source_matches:
        return None
    source = source_matches[-1]
    # Do not let an old source attribution leak across several complete
    # sentences into an unrelated answer event.
    if len(re.findall(r"[.!?\u3002\uff01\uff1f]", text[source.end() : start])) > 1:
        return None

    for retraction in _EXPLICIT_RETRACTION_EVENT_RE.finditer(
        text, source.end(), start
    ):
        if not _event_is_negated(text, start=retraction.start()):
            return None

    follow_end = min(len(text), max(end, start) + 240)
    window_start = max(0, source.start() - 300)
    window = text[window_start:follow_end]
    stance_events: list[tuple[int, int, str]] = []

    def in_source_scope(match: re.Match[str]) -> bool:
        absolute_end = window_start + match.end()
        if absolute_end > source.start():
            return True
        intervening = text[absolute_end : source.start()]
        return len(re.findall(r"[.!?\u3002\uff01\uff1f]", intervening)) <= 1

    if (
        _PERSONAL_COMPUTATION_SOURCE_RE.search(
            text, source.start(), min(len(text), start + 1)
        )
        is not None
        or _ADOPTED_COMPUTATION_SOURCE_RE.search(
            text, source.start(), min(len(text), start + 1)
        )
        is not None
    ):
        stance_events.append((source.start(), 1, "adopted"))
    stance_events.extend(
        (window_start + match.start(), 1, "adopted")
        for match in _SOURCE_ADOPTION_STANCE_RE.finditer(window)
        if in_source_scope(match)
    )
    stance_events.extend(
        (window_start + match.start(), 2, "rejected")
        for match in _SOURCE_REJECTION_STANCE_RE.finditer(window)
        if in_source_scope(match)
    )
    stance_events.extend(
        (window_start + match.start(), 3, "rejected")
        for match in _SOURCE_NEGATED_ADOPTION_STANCE_RE.finditer(window)
        if in_source_scope(match)
    )
    if not stance_events:
        return "reported"
    return max(stance_events, key=lambda event: (event[0], event[1]))[2]


def _evidence_match_is_contextual(
    text: str,
    *,
    start: int,
    end: int,
    matched_text: str,
) -> bool:
    """Reject quoted, hypothetical, or other-answer evidence."""

    source_stance = _source_evidence_stance(text, start=start, end=end)
    if _position_is_quoted(text, end):
        return source_stance != "adopted"
    boundary = max(
        text.rfind(mark, 0, start) for mark in (".", "!", "?", ";", "\n")
    )
    local_context = text[boundary + 1 : end]
    if _CONDITIONAL_ANSWER_SUFFIX_RE.search(local_context):
        return True
    if _CONTEXTUAL_EVIDENCE_SOURCE_RE.search(local_context):
        if _EXPLICIT_RETRACTION_EVENT_RE.search(matched_text) is not None:
            return False
        return source_stance != "adopted"
    if _INTERVENING_RETRACTION_ANTECEDENT_RE.search(local_context):
        return True

    if re.match(
        r"(?is)\s*(?:that\s+answer|the\s+(?:previous|earlier|above)\s+answer)\b",
        matched_text,
    ):
        previous_boundary = max(
            text.rfind(mark, 0, max(0, boundary))
            for mark in (".", "!", "?", ";", "\n")
        )
        previous_clause = text[previous_boundary + 1 : boundary + 1]
        if _INTERVENING_RETRACTION_ANTECEDENT_RE.search(previous_clause):
            return True
    return False


def _replacement_event_payload(
    text: str,
    *,
    cue_start: int,
    cue_end: int,
) -> tuple[bool, str, int]:
    """Return ``(is_answer_event, payload, end)`` for a revision cue.

    Discourse words such as ``however`` and ``therefore`` are common in normal
    reasoning, so the cue alone is not an answer event.  It becomes one only
    when its local clause starts with a concrete answer, an explicit answer
    predicate (``the correct value is`` / ``it is``), or ends with an adoption
    predicate (``55 is correct``).  The returned scoring payload excludes all
    earlier prose and the replacement cue itself.
    """

    end = _candidate_clause_end(text, cue_end)
    body = text[cue_end:end].strip(" \t\r\n:=,;`*_")
    # Let the ordinary explicit-cue parser retain its richer scoring text.
    # `_answer_cue_strength` promotes it to a committed event because the
    # replacement discourse marker is immediately before it.
    if _EXPLICIT_ANSWER_CUE_RE.match(body) is not None:
        return False, "", end
    explicit_prefix = _REPLACEMENT_ANSWER_PREFIX_RE.match(body)
    if explicit_prefix is not None:
        payload = body[explicit_prefix.end() :].strip()
        return True, payload.rstrip(".!\u3002\uff01").strip(), end

    correct_suffix = _REPLACEMENT_CORRECT_SUFFIX_RE.fullmatch(body)
    if correct_suffix is not None:
        payload = correct_suffix.group("payload").strip()
        return True, payload.rstrip(".!\u3002\uff01").strip(), end

    if _CONCRETE_MATH_ANSWER_PREFIX_RE.match(body) is None:
        return False, "", end
    payload = body.rstrip(".!\u3002\uff01").strip()
    return True, payload, end


def _event_is_negated(text: str, *, start: int) -> bool:
    """Return whether an apparent authority event is under explicit negation."""

    clause_start = max(
        text.rfind(mark, 0, start) for mark in (".", "!", "?", ";", "\n")
    )
    prefix = text[clause_start + 1 : start]
    return _NEGATED_EVENT_SCOPE_RE.search(prefix) is not None


def _answer_authority_boundaries(text: str) -> list[int]:
    """Return non-contextual correction/retraction boundary offsets."""

    boundaries: set[int] = set()
    for match in _EXPLICIT_CORRECTION_MARKER_RE.finditer(text):
        marker = re.search(
            r"(?is)\b(?:correction|corrected\s+answer|revised\s+answer)\b",
            match.group(0),
        )
        start = match.start() + (marker.start() if marker is not None else 0)
        if (
            not _event_is_negated(text, start=start)
            and not _evidence_match_is_contextual(
                text,
                start=start,
                end=match.end(),
                matched_text=match.group(0),
            )
        ):
            boundaries.add(start)
    for cue in _RETRACTION_CUE_START_RE.finditer(text):
        relative = _RETRACTED_ANSWER_PREFIX_RE.match(text[cue.start() :])
        if relative is None:
            continue
        end = cue.start() + relative.end()
        matched_text = text[cue.start() : end]
        if not _event_is_negated(
            text, start=cue.start()
        ) and not _evidence_match_is_contextual(
            text,
            start=cue.start(),
            end=end,
            matched_text=matched_text,
        ):
            boundaries.add(cue.start())
    for match in _CURRENT_ANSWER_RETRACTION_RE.finditer(text):
        if not _event_is_negated(
            text, start=match.start()
        ) and not _evidence_match_is_contextual(
            text,
            start=match.start(),
            end=match.end(),
            matched_text=match.group(0),
        ):
            boundaries.add(match.start())
    for match in _EXPLICIT_RETRACTION_EVENT_RE.finditer(text):
        if not _event_is_negated(
            text, start=match.start()
        ) and not _evidence_match_is_contextual(
            text,
            start=match.start(),
            end=match.end(),
            matched_text=match.group(0),
        ):
            boundaries.add(match.start())
    for match in _QUESTIONED_ANSWER_EVENT_RE.finditer(text):
        if not _event_is_negated(
            text, start=match.start()
        ) and not _evidence_match_is_contextual(
            text,
            start=match.start(),
            end=match.end(),
            matched_text=match.group(0),
        ):
            boundaries.add(match.start())
    for cue in _BARE_REPLACEMENT_CUE_RE.finditer(text):
        is_answer_event, _payload, end = _replacement_event_payload(
            text,
            cue_start=cue.start(),
            cue_end=cue.end(),
        )
        if (
            is_answer_event
            and not _event_is_negated(text, start=cue.start())
            and not _evidence_match_is_contextual(
                text,
                start=cue.start(),
                end=end,
                matched_text=text[cue.start() : end],
            )
        ):
            boundaries.add(cue.start())
    return sorted(boundaries)


def _answer_candidate_is_retracted(suffix: str) -> bool:
    """Return whether later text explicitly retracts the candidate."""

    return bool(_answer_authority_boundaries(suffix))


def _comparable_option_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    for spacing_command in (r"\,", r"\;", r"\:", r"\!", r"\quad", r"\qquad"):
        text = text.replace(spacing_command, "")
    # Option sources and generations routinely use equivalent LaTeX spellings
    # (``^2`` vs ``^{2}``, ``\\le`` vs ``\\leq``).  Normalize only cosmetic
    # syntax here, after the question has already been proven to contain a
    # structured option set; this never changes general free-response maths.
    text = re.sub(r"\\(le|ge)q\b", r"\\\1", text)
    text = re.sub(
        r"([_^])\s*\{\s*([+-]?(?:\d+|[A-Za-z]))\s*\}",
        r"\1\2",
        text,
    )
    text = text.strip("`*_ $\t\r\n")
    text = text.rstrip(".\u3002;:!? ")
    return _WHITESPACE_RE.sub("", text).casefold()


def _explicit_option_label(
    scoring_text: str,
    labels: set[str],
    *,
    recover_incomplete_tail: bool = True,
) -> str | None:
    """Extract only strong final-choice signals, never incidental prose labels."""

    window = _math_verify_input(
        scoring_text, recover_incomplete_tail=recover_incomplete_tail
    )
    # Some completions serialize their final answer inside a JSON/string
    # envelope, leaving ``\\boxed{A}`` double-escaped.  Within a question whose
    # option set is already proven, either one or multiple leading backslashes
    # still constitute an unambiguous final label.
    escaped_boxed = re.findall(
        r"\\+boxed\s*\{\s*([A-Z])\s*\}?", window, flags=re.IGNORECASE
    )
    escaped_boxed = [label.upper() for label in escaped_boxed]
    escaped_boxed = [label for label in escaped_boxed if label in labels]
    if escaped_boxed:
        return escaped_boxed[-1]
    boxed = _last_boxed_content(window)
    if boxed is not None:
        label = _reference_option_label(boxed)
        if label in labels:
            return label
        # A model may preserve the selected option's complete text inside the
        # box, for example ``\text{(D) } y=f(x)+11``.  Once the question has
        # proven the option set, a parenthesized label at the very beginning
        # of the boxed payload is authoritative rather than an incidental
        # letter in mathematical prose.
        prefixed = re.match(
            r"^(?:\\(?:text|mathrm|mathbf)\s*\{\s*)?"
            r"[\(\[]\s*([A-Z])\s*[\)\]]",
            boxed.strip(),
            flags=re.IGNORECASE,
        )
        if prefixed and prefixed.group(1).upper() in labels:
            return prefixed.group(1).upper()
    matches = [
        match.group(1).upper()
        for match in _EXPLICIT_OPTION_LABEL_RE.finditer(window)
        if match.group(1) in labels
    ]
    if matches:
        return matches[-1]
    lines = [line.strip() for line in window.splitlines() if line.strip()]
    if not lines:
        return None
    final_line = lines[-1].strip("`*_")
    bare = re.fullmatch(
        r"[\(\[]?\s*([A-Z])\s*[\)\]]?[.!]?", final_line, flags=re.IGNORECASE
    )
    if bare and bare.group(1).upper() in labels:
        return bare.group(1).upper()
    leading = re.match(
        r"^(?:\\+\s*)?[\(\[]\s*([A-Z])\s*[\)\]](?:\s+|$)",
        final_line,
        flags=re.IGNORECASE,
    )
    if leading and leading.group(1).upper() in labels:
        return leading.group(1).upper()
    return None


def _final_answer_candidates(
    scoring_text: str, *, recover_incomplete_tail: bool = True
) -> list[str]:
    window = _math_verify_input(
        scoring_text, recover_incomplete_tail=recover_incomplete_tail
    )
    candidates: list[str] = []
    boxed = _last_boxed_content(window)
    if boxed is not None:
        candidates.append(boxed)
    candidates.extend(match.group(1) for match in _FINAL_ANSWER_VALUE_RE.finditer(window))
    lines = [line.strip() for line in window.splitlines() if line.strip()]
    if lines and len(lines[-1]) <= 500:
        candidates.append(lines[-1])
    return candidates


def _short_text(value: str, *, limit: int = 1200) -> str:
    normalized = _normalize_text(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _normalize_answer_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized if normalized else None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        normalized = str(value)
        return normalized.strip() or None
    normalized = str(value).strip()
    return normalized or None


def resolve_reference_answer(record: FreeAnswerRecord) -> str:
    metadata = record.metadata or {}
    for key in _PREFERRED_ANSWER_KEYS:
        normalized = _normalize_answer_value(metadata.get(key))
        if normalized:
            return normalized
    raw_record = metadata.get("raw_record")
    if isinstance(raw_record, dict):
        for key in _PREFERRED_ANSWER_KEYS:
            normalized = _normalize_answer_value(raw_record.get(key))
            if normalized:
                return normalized
    return record.answer


def _iter_completions(source: Iterable[dict] | str | Path) -> Iterable[dict]:
    if isinstance(source, (str, Path)):
        yield from iter_jsonl(source)
        return
    yield from source


@dataclass(slots=True)
class FreeResponseEvaluation:
    metrics_by_group: dict[str, dict[str, float]]
    rows_by_group: dict[str, list[tuple[int, int, bool]]]
    samples: int
    payloads: list[dict]
    payloads_by_group: dict[str, list[dict]] = field(default_factory=dict)
    judge_stats_by_group: dict[str, dict[str, object]] = field(default_factory=dict)
    math_verify_retry_stats_by_group: dict[str, dict[str, object]] = field(
        default_factory=dict
    )
    primary_group: str = STRATEGY_A

    @property
    def exact_accuracy(self) -> float:
        return float(self.metrics_by_group.get(self.primary_group, {}).get("exact_accuracy", 0.0))

    @property
    def judge_accuracy(self) -> float | None:
        value = self.metrics_by_group.get(self.primary_group, {}).get("judge_accuracy")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def rows(self) -> list[tuple[int, int, bool]]:
        return list(self.rows_by_group.get(self.primary_group, []))


@dataclass(slots=True)
class LLMJudgeConfig:
    api_key: str
    model: str
    base_url: str | None = None
    timeout_s: float = 60.0
    max_workers: int = 4
    max_completion_tokens: int | None = None
    temperature: float = 0.0

    max_retries: int = 3
    backoff_base: float = 0.5
    recovery_rounds: int = 2

    prompt_template: str = DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE


def llm_judge_protocol(config: LLMJudgeConfig) -> dict[str, object]:
    """Describe the content-affecting Judge protocol without connection secrets.

    ``api_key`` and ``base_url`` are deliberately absent.  The returned
    mapping is safe to persist in score metrics and contains enough evidence
    for a replay monitor to prove that the deterministic temperature-zero
    request and the strict boolean response contract were used.
    """

    model = str(config.model)
    return {
        "protocol_version": LLM_JUDGE_PROTOCOL_VERSION,
        "model": model,
        "temperature": float(config.temperature),
        "prompt_template_sha256": llm_judge_prompt_sha256(config.prompt_template),
        "max_completion_tokens": (
            int(config.max_completion_tokens)
            if config.max_completion_tokens is not None
            else None
        ),
        "response_contract": LLM_JUDGE_RESPONSE_CONTRACT,
        "stream": False,
        "qwen3_enable_thinking": False if "qwen3" in model.lower() else None,
        "max_workers": int(config.max_workers),
        "max_retries": int(config.max_retries),
        "recovery_rounds": int(config.recovery_rounds),
    }


def _llm_judge_protocol_digest(protocol: dict[str, object]) -> str:
    canonical = json.dumps(
        protocol,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def llm_judge_protocol_fingerprint(config: LLMJudgeConfig) -> str:
    """Return the canonical SHA-256 fingerprint of :func:`llm_judge_protocol`."""

    return _llm_judge_protocol_digest(llm_judge_protocol(config))


def llm_judge_protocol_stats_reasons(
    stats: Mapping[str, object],
    *,
    expected_model: str | None = None,
    expected_prompt_template: str | None = None,
    expected_max_completion_tokens: int | None | object = ...,
    expected_max_workers: int | None = None,
) -> list[str]:
    """Validate persisted Judge evidence without trusting a deployment time.

    The fingerprint is recomputed from the persisted semantic fields, while
    the fixed protocol contract is checked independently.  Callers may also
    pin deployment-specific model, prompt, token budget, and worker count.
    """

    protocol_keys = (
        "protocol_version",
        "model",
        "temperature",
        "prompt_template_sha256",
        "max_completion_tokens",
        "response_contract",
        "stream",
        "qwen3_enable_thinking",
        "max_workers",
        "max_retries",
        "recovery_rounds",
    )
    reasons: list[str] = []
    missing = [key for key in protocol_keys if key not in stats]
    if missing:
        return ["judge_protocol_missing_fields:" + ",".join(missing)]
    protocol = {key: stats.get(key) for key in protocol_keys}
    fingerprint = str(stats.get("protocol_fingerprint_sha256") or "")
    computed = _llm_judge_protocol_digest(protocol)
    if fingerprint != computed:
        reasons.append(
            f"judge_protocol_fingerprint:{fingerprint or 'missing'}!=expected:{computed}"
        )
    if stats.get("protocol_version") != LLM_JUDGE_PROTOCOL_VERSION:
        reasons.append(
            f"judge_protocol_version:{stats.get('protocol_version')!r}"
            f"!=expected:{LLM_JUDGE_PROTOCOL_VERSION!r}"
        )
    try:
        temperature = float(stats.get("temperature"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        temperature = float("nan")
    if temperature != 0.0:
        reasons.append(f"judge_temperature:{stats.get('temperature')!r}!=expected:0.0")
    if stats.get("response_contract") != LLM_JUDGE_RESPONSE_CONTRACT:
        reasons.append(
            f"judge_response_contract:{stats.get('response_contract')!r}"
            f"!=expected:{LLM_JUDGE_RESPONSE_CONTRACT!r}"
        )
    if stats.get("stream") is not False:
        reasons.append(f"judge_stream:{stats.get('stream')!r}!=expected:False")
    model = str(stats.get("model") or "")
    if not model:
        reasons.append("judge_model_missing")
    elif expected_model and model != expected_model:
        reasons.append(f"judge_model:{model!r}!=expected:{expected_model!r}")
    expected_qwen_thinking = False if "qwen3" in model.lower() else None
    if stats.get("qwen3_enable_thinking") is not expected_qwen_thinking:
        reasons.append(
            f"judge_qwen3_enable_thinking:{stats.get('qwen3_enable_thinking')!r}"
            f"!=expected:{expected_qwen_thinking!r}"
        )
    prompt_hash = str(stats.get("prompt_template_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", prompt_hash):
        reasons.append("judge_prompt_template_sha256_invalid")
    if expected_prompt_template is not None:
        expected_prompt_hash = llm_judge_prompt_sha256(expected_prompt_template)
        if prompt_hash != expected_prompt_hash:
            reasons.append(
                f"judge_prompt_template_sha256:{prompt_hash or 'missing'}"
                f"!=expected:{expected_prompt_hash}"
            )
    if expected_max_completion_tokens is not ...:
        actual_tokens = stats.get("max_completion_tokens")
        if actual_tokens != expected_max_completion_tokens:
            reasons.append(
                f"judge_max_completion_tokens:{actual_tokens!r}"
                f"!=expected:{expected_max_completion_tokens!r}"
            )
    try:
        max_workers = int(stats.get("max_workers"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        max_workers = 0
    if max_workers < 1:
        reasons.append(f"judge_max_workers:{stats.get('max_workers')!r}!=positive")
    elif expected_max_workers is not None and max_workers != expected_max_workers:
        reasons.append(
            f"judge_max_workers:{max_workers}!=expected:{expected_max_workers}"
        )
    if stats.get("max_retries") != 3:
        reasons.append(f"judge_max_retries:{stats.get('max_retries')!r}!=expected:3")
    if stats.get("recovery_rounds") != 2:
        reasons.append(
            f"judge_recovery_rounds:{stats.get('recovery_rounds')!r}!=expected:2"
        )
    return reasons


@dataclass(slots=True)
class LLMJudgeStats:
    total: int = 0
    parsed_count: int = 0
    invalid_output_count: int = 0
    request_error_count: int = 0
    invalid_output_examples: list[str] = field(default_factory=list)
    request_error_examples: list[str] = field(default_factory=list)
    protocol: dict[str, object] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return self.invalid_output_count + self.request_error_count

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "total": self.total,
            "parsed_count": self.parsed_count,
            "invalid_output_count": self.invalid_output_count,
            "request_error_count": self.request_error_count,
            "error_count": self.error_count,
            "invalid_output_examples": self.invalid_output_examples,
            "request_error_examples": self.request_error_examples,
        }
        if self.protocol:
            payload.update(self.protocol)
            payload["protocol_fingerprint_sha256"] = _llm_judge_protocol_digest(
                self.protocol
            )
        return payload


class LLMJudge:
    def __init__(self, config: LLMJudgeConfig) -> None:
        self.config = config
        timeout = max(1.0, float(config.timeout_s))
        relay_base_url = os.environ.get("RWKV_JUDGE_RELAY_BASE_URL", "").strip()
        if not relay_base_url:
            relay_file = Path(
                os.environ.get(
                    "RWKV_JUDGE_RELAY_BASE_URL_FILE",
                    ".judge-relay-base-url",
                )
            )
            try:
                relay_base_url = relay_file.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                pass
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=relay_base_url or config.base_url,
            timeout=timeout,
            max_retries=0,
            # Judge routing must not inherit the benchmark host's generic
            # HTTP(S)_PROXY.  The inference runners use a LAN proxy for data
            # access, but the configured OpenAI-compatible judge endpoint is
            # independently reachable and a stale proxy would otherwise turn
            # every request failure into a false judgement.
            http_client=httpx.Client(trust_env=False, timeout=timeout),
        )
        self.last_run_stats: LLMJudgeStats | None = None

    def judge(self, items: list[tuple[str, str, str]]) -> list[bool]:
        """Return judge flags for (question, reference, prediction) items."""

        def worker(entry: tuple[str, str, str]) -> tuple[bool, str, str | None]:
            question, reference, prediction = entry
            prompt = self.config.prompt_template
            prompt = prompt.replace("<Q>", question)
            prompt = prompt.replace("<REF>", reference)
            prompt = prompt.replace("<A>", prediction)

            last_error = ""
            last_error_kind = "request_error"
            for attempt in range(self.config.max_retries + 1):
                try:
                    request_kwargs: dict[str, Any] = {
                        "model": self.config.model,
                        "stream": False,
                        "temperature": self.config.temperature,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if "qwen3" in self.config.model.lower():
                        request_kwargs["extra_body"] = {
                            "chat_template_kwargs": {"enable_thinking": False}
                        }
                    if self.config.max_completion_tokens is not None:
                        request_kwargs["max_tokens"] = self.config.max_completion_tokens
                    response = self.client.chat.completions.create(**request_kwargs)
                    content = (response.choices[0].message.content or "").strip()

                    if content not in {"True", "False"}:
                        last_error_kind = "invalid_output"
                        last_error = f"invalid output: {content!r}"
                        raise ValueError(f"LLM judge 输出非法值: {content!r}")

                    return content == "True", "parsed", None

                except Exception as exc:
                    if not last_error:
                        last_error = repr(exc)
                    if last_error_kind != "invalid_output":
                        last_error_kind = "request_error"
                    if attempt == self.config.max_retries:
                        return False, last_error_kind, last_error

                    backoff = self.config.backoff_base * (2**attempt)
                    time.sleep(backoff)

            return False, last_error_kind, last_error or None

        results: list[bool] = [False for _ in range(len(items))]
        statuses: list[str] = ["request_error" for _ in range(len(items))]
        details: list[str | None] = [None for _ in range(len(items))]

        def run_indices(indices: list[int], *, desc: str) -> None:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(worker, items[idx]): idx for idx in indices
                }
                for future in tqdm.tqdm(
                    as_completed(futures), total=len(futures), desc=desc
                ):
                    idx = futures[future]
                    passed, status, detail = future.result()
                    results[idx] = passed
                    statuses[idx] = status
                    details[idx] = detail

        run_indices(list(range(len(items))), desc="LLM judging")

        # A large judge batch can finish with only a handful of transient
        # transport failures even after each request's normal retries.  Do a
        # small number of recovery sweeps over only those failed entries so a
        # 4k-completion evaluation does not have to repeat every successful
        # judge call.  Statistics below describe the final outcomes only.
        for recovery_round in range(max(0, self.config.recovery_rounds)):
            pending = [idx for idx, status in enumerate(statuses) if status != "parsed"]
            if not pending:
                break
            time.sleep(self.config.backoff_base * (2**recovery_round))
            run_indices(
                pending,
                desc=f"LLM judge recovery {recovery_round + 1}",
            )

        stats = LLMJudgeStats(
            total=len(items),
            protocol=llm_judge_protocol(self.config),
        )
        for status, detail in zip(statuses, details, strict=True):
            if status == "parsed":
                stats.parsed_count += 1
            elif status == "invalid_output":
                stats.invalid_output_count += 1
                if detail and len(stats.invalid_output_examples) < 5:
                    stats.invalid_output_examples.append(detail)
            else:
                stats.request_error_count += 1
                if detail and len(stats.request_error_examples) < 5:
                    stats.request_error_examples.append(detail)
        self.last_run_stats = stats
        return results


@dataclass(slots=True)
class _MathVerifyResult:
    passed: bool
    answer: str
    fail_reason: str


@dataclass(frozen=True, slots=True)
class _DeterministicMathVerifyOutcome:
    passed: bool
    limit_exceeded: bool = False
    attempted_bijections: int = 0


@dataclass(slots=True)
class _MultipleChoiceVerifyResult:
    result: _MathVerifyResult
    conclusive: bool


@dataclass(slots=True)
class _ScoredCompletion:
    source_payload: dict[str, Any]
    sample_index: int
    repeat_index: int
    question: str
    reference: str
    scoring_text: str
    display_answer: str
    math_passed: bool
    final_passed: bool
    fail_reason: str
    judge_eligible: bool = True


@lru_cache(maxsize=1)
def _load_math_verify() -> tuple[Callable[..., Any], Callable[..., Any]] | None:
    try:
        from math_verify import parse as raw_parse
        from math_verify import verify as raw_verify
    except ImportError:
        return None

    def parse(value: str) -> Any:
        # The scorer owns one outer deadline.  math-verify's nested SIGALRM
        # catches its own timeout and turns it into an ordinary false result,
        # which makes a transient timeout indistinguishable from a mismatch.
        return raw_parse(value, parsing_timeout=None, raise_on_error=True)

    def verify(gold: Any, pred: Any, *, strict: bool = True) -> bool:
        if strict is not True:
            raise ValueError(
                "non-strict math-verify is nondeterministic; use the project "
                "symbol-bijection verifier"
            )
        with _deterministic_math_verify_grader_runtime():
            return bool(
                raw_verify(
                    gold,
                    pred,
                    strict=strict,
                    timeout_seconds=None,
                    raise_on_error=True,
                )
            )

    return parse, verify


def _parsed_extractions(parsed: Any) -> tuple[Any, ...]:
    """Mirror math-verify's list-or-singleton extraction contract."""

    return tuple(parsed) if isinstance(parsed, list) else (parsed,)


def _stable_symbol_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized = (
            (
                _stable_symbol_metadata(key),
                _stable_symbol_metadata(item),
            )
            for key, item in value.items()
        )
        return tuple(sorted(normalized, key=repr))
    if isinstance(value, (set, frozenset)):
        normalized = (_stable_symbol_metadata(item) for item in value)
        return tuple(sorted(normalized, key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_stable_symbol_metadata(item) for item in value)
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    value_type = type(value)
    return (
        value_type.__module__,
        value_type.__qualname__,
        repr(value),
    )


def _stable_symbol_attribute_key(symbol: Any, attribute: str) -> str:
    try:
        value = getattr(symbol, attribute)
    except _MathVerifyTimeout:
        raise
    except Exception:  # noqa: BLE001
        return "<unavailable>"
    return repr(_stable_symbol_metadata(value))


def _symbol_sort_key(symbol: Any) -> tuple[str, ...]:
    """Return a process-stable order for SymPy free symbols."""

    symbol_type = type(symbol)
    return (
        _stable_symbol_attribute_key(symbol, "name"),
        symbol_type.__module__,
        symbol_type.__qualname__,
        _stable_symbol_attribute_key(symbol, "_assumptions0"),
        _stable_symbol_attribute_key(symbol, "shape"),
        repr(_stable_symbol_metadata(symbol)),
    )


def _stable_free_symbols(expression: Any) -> tuple[Any, ...] | None:
    try:
        ordered = tuple(sorted(expression.free_symbols, key=_symbol_sort_key))
    except (AttributeError, TypeError):
        return None
    for previous, current in zip(ordered, ordered[1:]):
        if _symbol_sort_key(previous) == _symbol_sort_key(current):
            # A stable key collision would reintroduce the source set's
            # process-dependent iteration order.  Refuse that extraction pair.
            return None
    return ordered


def _symbols_are_bijection_compatible(
    pred_symbol: Any,
    gold_symbol: Any,
) -> bool | None:
    """Return whether two symbols may be renamed, or ``None`` if unknowable."""

    try:
        pred_commutative = getattr(pred_symbol, "is_commutative")
        gold_commutative = getattr(gold_symbol, "is_commutative")
        pred_shape = getattr(pred_symbol, "shape", _MISSING_SYMBOL_ATTRIBUTE)
        gold_shape = getattr(gold_symbol, "shape", _MISSING_SYMBOL_ATTRIBUTE)
    except _MathVerifyTimeout:
        raise
    except Exception:  # noqa: BLE001
        return None
    if pred_commutative != gold_commutative:
        return False
    if (pred_shape is _MISSING_SYMBOL_ATTRIBUTE) != (
        gold_shape is _MISSING_SYMBOL_ATTRIBUTE
    ):
        return False
    if pred_shape is not _MISSING_SYMBOL_ATTRIBUTE and (
        _stable_symbol_metadata(pred_shape) != _stable_symbol_metadata(gold_shape)
    ):
        return False
    return True


@contextmanager
def _deterministic_math_verify_grader_runtime() -> Any:
    """Stabilize math-verify's relational ``solve(..., free_symbols)`` fallback."""

    from math_verify import grader as math_verify_grader

    with _MATH_VERIFY_GRADER_LOCK:
        original_solve = math_verify_grader.solve

        def ordered_solve(expression: Any, *symbols: Any, **kwargs: Any) -> Any:
            ordered_symbols = tuple(
                tuple(sorted(item, key=_symbol_sort_key))
                if isinstance(item, (set, frozenset))
                else item
                for item in symbols
            )
            return original_solve(expression, *ordered_symbols, **kwargs)

        math_verify_grader.solve = ordered_solve
        try:
            yield
        finally:
            math_verify_grader.solve = original_solve


def _deterministic_symbol_mapping_parts(
    gold_symbols: tuple[Any, ...],
    pred_symbols: tuple[Any, ...],
) -> tuple[dict[Any, Any], tuple[Any, ...], tuple[Any, ...]] | None:
    """Fix same-name symbols first and return deterministically ordered remainders."""

    gold_by_name: dict[str, list[Any]] = {}
    pred_by_name: dict[str, list[Any]] = {}
    for symbol in gold_symbols:
        gold_by_name.setdefault(str(getattr(symbol, "name", "")), []).append(symbol)
    for symbol in pred_symbols:
        pred_by_name.setdefault(str(getattr(symbol, "name", "")), []).append(symbol)

    mapping: dict[Any, Any] = {}
    matched_gold: set[Any] = set()
    matched_pred: set[Any] = set()
    for name in sorted(gold_by_name.keys() & pred_by_name.keys()):
        same_name_gold = sorted(gold_by_name[name], key=_symbol_sort_key)
        same_name_pred = sorted(pred_by_name[name], key=_symbol_sort_key)
        for pred_symbol, gold_symbol in zip(same_name_pred, same_name_gold):
            if not _symbols_are_bijection_compatible(pred_symbol, gold_symbol):
                return None
            mapping[pred_symbol] = gold_symbol
            matched_pred.add(pred_symbol)
            matched_gold.add(gold_symbol)

    remaining_gold = tuple(
        symbol for symbol in gold_symbols if symbol not in matched_gold
    )
    remaining_pred = tuple(
        symbol for symbol in pred_symbols if symbol not in matched_pred
    )
    return mapping, remaining_gold, remaining_pred


def _normalized_relation(expression: Any) -> tuple[str, Any] | None:
    """Normalize a SymPy relation to ``kind`` and a left-minus-right residual."""

    try:
        from sympy.core.relational import (
            Equality,
            GreaterThan,
            LessThan,
            StrictGreaterThan,
            StrictLessThan,
            Unequality,
        )
    except ImportError:
        return None

    if isinstance(expression, Equality):
        return "eq", expression.lhs - expression.rhs
    if isinstance(expression, Unequality):
        return "ne", expression.lhs - expression.rhs
    if isinstance(expression, StrictLessThan):
        return "lt", expression.lhs - expression.rhs
    if isinstance(expression, StrictGreaterThan):
        return "lt", expression.rhs - expression.lhs
    if isinstance(expression, LessThan):
        return "le", expression.lhs - expression.rhs
    if isinstance(expression, GreaterThan):
        return "le", expression.rhs - expression.lhs
    return None


def _deterministic_relational_equivalence(
    gold_expression: Any,
    pred_expression: Any,
) -> bool | None:
    """Compare relations up to a proven non-zero constant scale factor."""

    gold_relation = _normalized_relation(gold_expression)
    pred_relation = _normalized_relation(pred_expression)
    if gold_relation is None or pred_relation is None:
        return None
    gold_kind, gold_residual = gold_relation
    pred_kind, pred_residual = pred_relation
    if gold_kind != pred_kind:
        return False

    try:
        from sympy import cancel, simplify

        if simplify(gold_residual - pred_residual) == 0:
            return True
        gold_is_zero = gold_residual.is_zero
        pred_is_zero = pred_residual.is_zero
        if gold_is_zero is True or pred_is_zero is True:
            return False
        scale = simplify(cancel(gold_residual / pred_residual))
        if scale.free_symbols or scale.is_zero is not False:
            return False
        if scale.is_finite is False:
            return False
        if gold_kind in {"lt", "le"}:
            return scale.is_positive is True
        return True
    except _MathVerifyTimeout:
        raise
    except Exception:  # noqa: BLE001
        return False


def _deterministic_math_verify(
    gold: Any,
    pred: Any,
    verify: Callable[..., Any],
    *,
    max_bijections: int = _MAX_DETERMINISTIC_SYMBOL_BIJECTIONS,
) -> _DeterministicMathVerifyOutcome:
    """Verify parsed answers with deterministic, simultaneous symbol renaming.

    math-verify's non-strict path zips two unordered ``free_symbols`` sets and
    applies sequential ``subs`` calls.  Both choices can change the result with
    ``PYTHONHASHSEED``.  We instead keep shared names fixed, enumerate the
    remaining bijections in a stable order, rename simultaneously with
    ``xreplace``, and compare every candidate strictly.  The global attempt cap
    prevents factorial work; reaching it fails closed rather than guessing.
    """

    if bool(verify(gold, pred, strict=True)):
        return _DeterministicMathVerifyOutcome(passed=True)

    attempted = 0
    for gold_expr in _parsed_extractions(gold):
        for pred_expr in _parsed_extractions(pred):
            gold_symbols = _stable_free_symbols(gold_expr)
            pred_symbols = _stable_free_symbols(pred_expr)
            if gold_symbols is None or pred_symbols is None:
                continue
            if not gold_symbols or len(gold_symbols) != len(pred_symbols):
                continue

            mapping_parts = _deterministic_symbol_mapping_parts(
                gold_symbols, pred_symbols
            )
            if mapping_parts is None:
                continue
            fixed, remaining_gold, remaining_pred = mapping_parts
            if len(remaining_gold) != len(remaining_pred):
                continue
            for gold_permutation in permutations(remaining_gold):
                if attempted >= max_bijections:
                    return _DeterministicMathVerifyOutcome(
                        passed=False,
                        limit_exceeded=True,
                        attempted_bijections=attempted,
                    )
                attempted += 1
                compatible = tuple(
                    _symbols_are_bijection_compatible(pred_symbol, gold_symbol)
                    for pred_symbol, gold_symbol in zip(
                        remaining_pred,
                        gold_permutation,
                        strict=True,
                    )
                )
                if not all(item is True for item in compatible):
                    continue
                mapping = dict(fixed)
                mapping.update(zip(remaining_pred, gold_permutation, strict=True))
                try:
                    renamed_pred = pred_expr.xreplace(mapping)
                except (AttributeError, TypeError, ValueError):
                    continue
                relational_result = _deterministic_relational_equivalence(
                    gold_expr,
                    renamed_pred,
                )
                if relational_result is True:
                    return _DeterministicMathVerifyOutcome(
                        passed=True,
                        attempted_bijections=attempted,
                    )
                if bool(verify(gold_expr, renamed_pred, strict=True)):
                    return _DeterministicMathVerifyOutcome(
                        passed=True,
                        attempted_bijections=attempted,
                    )

    return _DeterministicMathVerifyOutcome(
        passed=False,
        attempted_bijections=attempted,
    )


class _MathVerifyTimeout(TimeoutError):
    pass


class UnresolvedMathVerifyTimeoutError(RuntimeError):
    """Stop score persistence when deterministic verification is unresolved."""

    def __init__(
        self,
        *,
        group: str,
        sample_index: int,
        repeat_index: int,
        first_fail_reason: str,
        retry_fail_reason: str,
    ) -> None:
        self.group = group
        self.sample_index = sample_index
        self.repeat_index = repeat_index
        self.first_fail_reason = first_fail_reason
        self.retry_fail_reason = retry_fail_reason
        super().__init__(
            "unresolved math-verify timeout; refusing to persist an incorrect "
            f"score (group={group}, sample_index={sample_index}, "
            f"repeat_index={repeat_index}, first={first_fail_reason!r}, "
            f"retry={retry_fail_reason!r})"
        )


class UnresolvedMathVerifySymbolBijectionError(RuntimeError):
    """Block persistence when the deterministic equivalence search is incomplete."""

    def __init__(
        self,
        *,
        group: str,
        sample_index: int,
        repeat_index: int,
    ) -> None:
        self.group = group
        self.sample_index = sample_index
        self.repeat_index = repeat_index
        super().__init__(
            "math-verify symbol-bijection limit reached; refusing to persist "
            "an unresolved score "
            f"(group={group}, sample_index={sample_index}, "
            f"repeat_index={repeat_index}, "
            f"max_bijections={_MAX_DETERMINISTIC_SYMBOL_BIJECTIONS})"
        )


def _math_verify_timeout_s() -> float:
    raw = os.getenv("RWKV_MATH_VERIFY_TIMEOUT_S")
    if raw is None or not raw.strip():
        return _DEFAULT_MATH_VERIFY_TIMEOUT_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_MATH_VERIFY_TIMEOUT_S


_MATH_VERIFY_TIMEOUT_REASONS = frozenset(
    {
        "reference_parse_timeout",
        "prediction_parse_timeout",
        "math_verify_timeout",
    }
)


def _is_math_verify_timeout_reason(reason: object) -> bool:
    """Return whether a deterministic verifier result timed out.

    This deliberately recognizes only the three fail-closed timeout outcomes
    emitted by :func:`_math_verify`.  Parse errors and ordinary mismatches are
    real evaluation results and must not be disguised as transient timeouts.
    """

    return str(reason or "") in _MATH_VERIFY_TIMEOUT_REASONS


@contextmanager
def _temporary_math_verify_timeout(timeout_s: float) -> Any:
    """Temporarily override the verifier deadline for one isolated retry."""

    key = "RWKV_MATH_VERIFY_TIMEOUT_S"
    previous = os.environ.get(key)
    os.environ[key] = f"{timeout_s:g}"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _env_flag(name: str) -> bool:
    raw = os.getenv(name)
    return raw is not None and raw.strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def _math_verify_time_limit() -> Any:
    timeout_s = _math_verify_timeout_s()
    if timeout_s <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise _MathVerifyTimeout(f"math_verify timed out after {timeout_s:g}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
        signal.signal(signal.SIGALRM, previous_handler)


def _reference_expr(reference: str) -> str:
    if "\\boxed" in reference:
        return reference
    return f"$\\boxed{{{reference}}}$"


@dataclass(frozen=True, slots=True)
class _AnswerCandidate:
    start: int
    end: int
    scoring_text: str
    content: str
    strength: int
    conflicting: bool = False
    explicit: bool = False


@dataclass(frozen=True, slots=True)
class _AnswerAssignment:
    """One field in an authoritative, possibly compound answer block."""

    label: str
    schema: str | None
    rhs: str
    rhs_complete: bool
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _AnswerBoxNode:
    """One ``\\boxed`` wrapper in the recursive answer-value tree."""

    start: int
    content_start: int
    content_end: int
    end: int | None
    children: tuple["_AnswerBoxNode", ...] = ()


@dataclass(frozen=True, slots=True)
class _AnswerEvidenceScan:
    state: str
    text: str
    candidate: _AnswerCandidate | None = None
    contextual_only: bool = False


_ANSWER_EVIDENCE_NONE = "none"
_ANSWER_EVIDENCE_CANDIDATE = "candidate"
_ANSWER_EVIDENCE_INVALIDATED = "invalidated"
_COMMITTED_ANSWER_STRENGTH = 4
_AUTHORITATIVE_ANSWER_FAIL_REASONS = {
    "authoritative_answer_invalidated",
    "contextual_answer_only",
    _SYMBOL_BIJECTION_LIMIT_FAIL_REASON,
}


def _select_answer_candidate(
    candidates: Iterable[_AnswerCandidate],
) -> _AnswerCandidate | None:
    """Select the final answer event without letting strength reorder time.

    Once a completion contains an answer commitment (a line-level answer,
    final answer, correction, or explicit replacement), the last complete
    commitment in textual order is authoritative.  Strength remains useful
    only to disambiguate overlapping parses at the same offset.  When there is
    no commitment at all, retain the historical strength-first fallback so an
    early explicit box is not displaced by a later incidental calculation.
    """

    materialized = list(candidates)
    if not materialized:
        return None
    committed = [
        candidate
        for candidate in materialized
        if candidate.strength >= _COMMITTED_ANSWER_STRENGTH
    ]
    if committed:
        return max(
            committed,
            key=lambda candidate: (
                candidate.start,
                candidate.strength,
                candidate.end,
            ),
        )
    return max(
        materialized,
        key=lambda candidate: (
            candidate.strength,
            candidate.start,
            candidate.end,
        ),
    )


def _answer_delimiters_balanced(value: str) -> bool:
    """Conservatively reject visibly unfinished answer expressions.

    Round/square mixed pairs are accepted only when their interior contains a
    comma, covering standard half-open interval notation without accepting a
    truncated arithmetic expression such as ``(1/2``.  Escaped braces are
    literals rather than LaTeX grouping delimiters.
    """

    stack: list[tuple[str, int]] = []
    matching = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(value):
        if char in "{}" and index > 0 and value[index - 1] == "\\":
            continue
        if char in "([{":
            stack.append((char, index))
            continue
        if char not in matching:
            continue
        if not stack:
            return False
        opener, opener_index = stack[-1]
        if opener == matching[char]:
            stack.pop()
            continue
        if opener in "([" and char in ")]" and "," in value[opener_index + 1 : index]:
            stack.pop()
            continue
        return False
    return not stack


def _latex_brace_groups_balanced(value: str) -> bool:
    r"""Validate LaTeX grouping without owning enclosing display delimiters.

    An assignment RHS can legitimately end with the committed block's ``\]``
    or ``\)`` closer, so the per-field gate must not require those delimiters
    to open inside that field.  Braces, however, are local to LaTeX commands;
    an unfinished presentation wrapper such as ``\text{\mathbf{3}`` is always
    incomplete and must fail closed.
    """

    depth = 0
    for index, char in enumerate(value):
        if char not in "{}":
            continue
        if index > 0 and value[index - 1] == "\\":
            continue
        if char == "{":
            depth += 1
        elif depth == 0:
            return False
        else:
            depth -= 1
    return depth == 0


def _without_latex_spacing(value: str) -> str:
    """Remove layout-only LaTeX commands before completeness checks."""

    return _LATEX_SPACING_COMMAND_RE.sub("", value)


def _latex_script_groups_are_complete(value: str) -> bool:
    """Reject empty/incomplete superscript and subscript annotations.

    Generated boxes sometimes stop after an underbrace annotation such as
    ``_{= \\; \\;}``. Its braces are technically balanced, but it contains no
    value and must not outrank an earlier real answer. This check applies to
    every simple script group, so it is structural rather than benchmark-
    specific.
    """

    compact = _without_latex_spacing(value)
    for match in _LATEX_SCRIPT_GROUP_RE.finditer(compact):
        content = match.group(1).strip()
        if not content or _TRAILING_INCOMPLETE_OPERATOR_RE.search(content):
            return False
    return True


def _answer_candidate_is_complete(value: str) -> bool:
    text = value.strip()
    if not _latex_required_arguments_are_semantically_complete(text):
        return False
    # Inspect the untrimmed authority block first.  A terminal period can be
    # either sentence punctuation or an empty ordered-field marker.  Once a
    # prior populated sibling establishes the ordered schema, trimming that
    # period would silently turn ``1. value; 2.`` into a complete scalar and
    # let the first value escape.
    raw_assignment_fields = _lex_authoritative_answer_assignments(text)
    preserve_terminal_order_marker = (
        _answer_block_has_incomplete_named_component_tail(text)
    )
    terminal_escaped_punctuation = text.endswith((r"\!", r"\."))
    if not text or text.endswith(("?", "？")):
        return False
    if not preserve_terminal_order_marker:
        text = text.rstrip(".!。！").rstrip()
    if terminal_escaped_punctuation:
        text = value.strip()
    # Paired Markdown emphasis at the end is presentation, while a lone ``*``
    # remains an unfinished multiplication operator.
    text = re.sub(
        r"(?:\*{2,}|_{2,}|(?<!`)`{1,2})$",
        "",
        text,
    ).rstrip()
    if re.search(r"(?<!\\)\*(?=\S)[^*\n]+(?<=\S)\*$", text):
        text = text[:-1].rstrip()
    # Ordered-field markers such as ``i)`` are presentation, but the generic
    # delimiter checker quite correctly sees their closing parenthesis as
    # unmatched.  Once the shared lexer recognizes an assignment block, use
    # its canonical representation for every subsequent structural check.
    canonical_assignments = _normalize_authoritative_assignment_block(text)
    assignment_fields = (
        raw_assignment_fields
        if preserve_terminal_order_marker
        else _lex_authoritative_answer_assignments(text)
    )
    if len(
        [assignment for assignment in assignment_fields if assignment.schema is not None]
    ) >= 2:
        text = canonical_assignments
    semantic_text = _semantic_answer_value_text(text).strip()
    if not semantic_text.strip("$*_`;,|&").strip():
        return False
    if r"\boxed" in text and not _answer_boxes_are_semantically_complete(
        text,
        allow_terminal_unclosed=True,
    ):
        return False
    if not text or not _answer_delimiters_balanced(text):
        return False
    _structure_spans, structure_is_open = _answer_structure_spans(text)
    if structure_is_open:
        return False
    # A closed box does not make a named multi-component answer complete when
    # the same committed block ends by introducing another component without
    # its value (``x=\\boxed{2}\ny=``).  Use the same normalized block
    # representation as continuation/conflict handling so the first component
    # cannot escape as a standalone answer.
    if _answer_block_has_incomplete_named_component_tail(text):
        return False
    compact = _without_latex_spacing(text).strip()
    if not compact or not _latex_script_groups_are_complete(compact):
        return False
    # An unmatched display-math delimiter is a reliable truncation signal.
    # Classify currency locally while scanning left-to-right: only an unsigned
    # scalar following an explicit money cue is currency.  Thus ``cost $2`` is
    # prose, while ``$-2``/``$+2`` and a bare ``answer is $2`` remain an
    # unclosed maths span.  Currency inside an already-open span does not close
    # that span, which also rejects ``$12 + ... and each item costs $2``.
    dollar_positions = [
        index
        for index, char in enumerate(text)
        if char == "$" and (index == 0 or text[index - 1] != "\\")
    ]
    money_context = re.compile(
        r"(?ix)(?:"
        r"cost(?:s|ing)?|price(?:d)?|pay(?:s|ing|ment)?|paid|spent|spend(?:s|ing)?|"
        r"worth|fee|charge(?:s|d)?|budget|salary|earn(?:s|ed|ing)?|"
        r"total(?:\s+(?:is|was|of))?|amount(?:\s+(?:is|was|of))?|"
        r"dollars?|usd"
        r")[^\n$]{0,32}$"
    )
    math_dollar_open = False
    for position in dollar_positions:
        numeric = re.match(
            r"\s*(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)",
            text[position + 1 :],
        )
        is_currency = False
        if numeric is not None and money_context.search(text[:position]):
            after_numeric = position + 1 + numeric.end()
            while after_numeric < len(text) and text[after_numeric].isspace():
                after_numeric += 1
            next_char = text[after_numeric] if after_numeric < len(text) else ""
            is_currency = not next_char or next_char not in "$+-*/^_=\\{}"
        if not is_currency:
            math_dollar_open = not math_dollar_open
    if math_dollar_open:
        return False
    if _TRAILING_INCOMPLETE_OPERATOR_RE.search(compact):
        return False
    if _TRAILING_CONTINUATION_RE.search(compact):
        return False
    return bool(re.search(r"[\w\d\\]", compact, re.UNICODE))


def _boxed_answer_candidates(text: str) -> list[_AnswerCandidate]:
    """Return complete boxed values, including a terminal pre-opened box.

    A final-stage template commonly contributes ``\\boxed{`` while generation
    contributes only the value.  Such a terminal box remains authoritative if
    its *content* is syntactically complete.  Conversely, an incomplete value
    (for example ``\\boxed{(-2)^``) is skipped so an earlier complete explicit
    answer can be considered.
    """

    candidates: list[_AnswerCandidate] = []
    final_answer_headings = list(_FINAL_ANSWER_HEADING_RE.finditer(text))

    def candidate_strength(start: int) -> int:
        """Promote boxes presented under a nearby ``Final Answer`` heading.

        Models often render a Markdown heading followed by several labelled
        subproblem values.  Such boxes are final-answer evidence, not ordinary
        reasoning boxes.  Restrict promotion to the structurally bounded
        heading block so later explanatory boxes do not acquire authority.
        """

        preceding = [heading for heading in final_answer_headings if heading.end() <= start]
        if not preceding:
            return 2
        heading = preceding[-1]
        scope_end = _explicit_answer_block_end(text, heading.end())
        return 4 if start <= scope_end else 2

    # Nested boxes are one recursive value, not independent answer identities.
    # Emit only roots; the recursive semantic gate validates every descendant.
    for node in _answer_box_forest(text):
        if not _answer_box_node_has_complete_semantic_value(
            text,
            node,
            allow_unclosed=node.end is None,
        ):
            continue
        end = node.end if node.end is not None else len(text)
        content = text[node.content_start : node.content_end]
        candidates.append(
            _AnswerCandidate(
                node.start,
                end,
                text[node.start:end],
                content,
                candidate_strength(node.start),
            )
        )
    return candidates


def _candidate_clause_end(text: str, cue_end: int, *, max_chars: int = 700) -> int:
    limit = min(len(text), cue_end + max_chars)
    cursor = cue_end
    # ``Final answer:\n\\boxed{...}`` is a common layout.  Do not end the
    # clause on the first newline when the cue has no payload yet.
    skipped_empty_line = False
    while cursor < limit:
        if text.startswith("</think>", cursor):
            return cursor
        char = text[cursor]
        if char in "\r\n":
            payload = text[cue_end:cursor].strip(" \t:=")
            if not payload and not skipped_empty_line:
                skipped_empty_line = True
                cursor += 1
                if char == "\r" and cursor < limit and text[cursor] == "\n":
                    cursor += 1
                continue
            next_start = cursor + 1
            if char == "\r" and next_start < limit and text[next_start] == "\n":
                next_start += 1
            next_end = next_start
            while next_end < limit and text[next_end] not in "\r\n":
                next_end += 1
            accumulated = text[cue_end:cursor]
            current_line = accumulated.rsplit("\n", 1)[-1]
            next_line = text[next_start:next_end]
            normalized_next = _normalize_answer_block_line(next_line)
            next_assignments = _lex_authoritative_answer_assignments(
                normalized_next
            )
            next_has_structured_lead = bool(
                _answer_block_structure_is_open(accumulated)
                or _line_is_answer_table_presentation(normalized_next)
                or _line_contains_only_boxed_answers(normalized_next)
                or _line_is_semantic_answer_continuation(normalized_next)
                or _line_is_named_answer_component(
                    normalized_next,
                    allow_group_prefix=True,
                )
                or any(
                    assignment.schema is not None
                    for assignment in next_assignments
                )
                or re.match(
                    r"^(?:\\(?:begin|end)\b|[()\[\]{}]|\\[{}])",
                    normalized_next,
                )
            )
            if next_has_structured_lead and _line_continues_explicit_answer_block(
                accumulated, current_line, next_line
            ):
                cursor = next_start
                continue
            return cursor
        if char in ".!?。！？":
            next_char = text[cursor + 1] if cursor + 1 < len(text) else ""
            if not next_char or next_char.isspace():
                # A period can be an ordered-field marker rather than a
                # sentence terminator: ``Final Answer: 1. x=...; 2. y=...``.
                # Keep scanning only when the marker is immediately followed
                # by an assignment-shaped field.  A scalar ``Answer: 2.`` and
                # ordinary prose therefore retain their normal clause end.
                ordered_prefix = text[cue_end : cursor + 1]
                ordered_suffix = text[cursor + 1 : limit]
                if char == "." and re.search(
                    r"(?is)(?:^|[;,\[(]\s*)"
                    r"(?:(?:\*{1,3}|_{1,3}|`{1,3}|~~)\s*)?"
                    r"(?:\d+|[ivxlcdm\u2160-\u2188]+)\.$",
                    ordered_prefix,
                ) and re.match(
                    r"(?is)^\s*(?:(?:\*{1,3}|_{1,3}|`{1,3}|~~)\s*)?"
                    r"(?:\\boxed\s*\{|"
                    r"(?:\\(?:mathbf|vec|boldsymbol|overrightarrow|mathrm)\s*\{\s*)?"
                    r"(?:[A-Za-z]\w*|\\[A-Za-z]+)"
                    r"(?:\s*[_^]\s*(?:\{[^{}]*\}|\w+))*\s*\}?\s*"
                    r"(?:&\s*)?(?::=|=|:|\bis\b))",
                    ordered_suffix,
                ):
                    cursor += 1
                    continue
                return cursor + 1
        cursor += 1
    return limit


def _explicit_answer_uses_block_layout(
    text: str,
    *,
    cue_start: int,
    cue_end: int,
) -> bool:
    """Return whether an explicit answer cue introduces a physical-line block.

    ``_EXPLICIT_ANSWER_CUE_RE`` may consume the newline following ``Answer:``.
    Inspect the cue's original physical line instead of relying on ``cue_end``
    alone, and require the cue itself to occupy the presentation-only prefix
    of its line.  This prevents an incidental ``the answer is`` in prose from
    swallowing later paragraphs.
    """

    line_start = text.rfind("\n", 0, cue_start) + 1
    line_end = text.find("\n", cue_start)
    if line_end < 0:
        line_end = len(text)
    effective_end = min(cue_end, line_end)
    if text[effective_end:line_end].strip():
        return False
    line_prefix = text[line_start:cue_start]
    return not _normalize_answer_block_line(
        line_prefix,
        presentation_prefix=True,
    )


def _answer_structure_spans(value: str) -> tuple[list[tuple[int, int]], bool]:
    """Return closed compound spans and whether a math wrapper remains open.

    This single structural scanner is shared by block continuation and
    compound-vs-independent answer classification.  It understands ordinary
    delimiters, half-open intervals, escaped set braces, LaTeX environments,
    and angle-vector delimiters.
    """

    spans: list[tuple[int, int]] = []
    delimiter_stack: list[tuple[str, int]] = []
    environment_stack: list[tuple[str, int]] = []
    matching = {")": "(", "]": "[", "}": "{"}
    display_stack: list[str] = []
    display_invalid = False
    index = 0
    while index < len(value):
        if value.startswith("```", index):
            line_start = value.rfind("\n", 0, index) + 1
            line_end = value.find("\n", index)
            if line_end < 0:
                line_end = len(value)
            fence_line = value[line_start:line_end].strip()
            if re.fullmatch(r"`{3,}(?:(?:latex|math)\s*)?", fence_line, re.I):
                if display_stack and display_stack[-1] == "<fence>":
                    display_stack.pop()
                else:
                    display_stack.append("<fence>")
                index = line_end
                continue
        display_token = next(
            (
                (token, kind, is_open)
                for token, kind, is_open in (
                    (r"\[", "<display-square>", True),
                    (r"\]", "<display-square>", False),
                    (r"\(", "<display-round>", True),
                    (r"\)", "<display-round>", False),
                )
                if value.startswith(token, index)
            ),
            None,
        )
        if display_token is not None:
            token, kind, is_open = display_token
            if is_open:
                display_stack.append(kind)
            elif display_stack and display_stack[-1] == kind:
                display_stack.pop()
            else:
                display_invalid = True
            index += len(token)
            continue
        if value.startswith("$$", index):
            if display_stack and display_stack[-1] == "<display-dollar>":
                display_stack.pop()
            else:
                display_stack.append("<display-dollar>")
            index += 2
            continue
        begin = re.match(r"\\begin\s*\{([^{}]+)\}", value[index:])
        if begin is not None:
            environment_stack.append((begin.group(1).strip(), index))
            index += begin.end()
            continue
        end = re.match(r"\\end\s*\{([^{}]+)\}", value[index:])
        if end is not None:
            name = end.group(1).strip()
            if environment_stack and environment_stack[-1][0] == name:
                _name, opener_index = environment_stack.pop()
                spans.append((opener_index, index + end.end()))
            index += end.end()
            continue
        angle_open = re.match(r"(?:\\left\s*)?\\langle\b", value[index:])
        if angle_open is not None:
            delimiter_stack.append(("<angle>", index))
            index += angle_open.end()
            continue
        angle_close = re.match(r"(?:\\right\s*)?\\rangle\b", value[index:])
        if angle_close is not None:
            if delimiter_stack and delimiter_stack[-1][0] == "<angle>":
                _opener, opener_index = delimiter_stack.pop()
                spans.append((opener_index, index + angle_close.end()))
            index += angle_close.end()
            continue
        if value.startswith(r"\{", index):
            delimiter_stack.append((r"\{", index))
            index += 2
            continue
        if value.startswith(r"\}", index):
            if delimiter_stack and delimiter_stack[-1][0] == r"\{":
                _opener, opener_index = delimiter_stack.pop()
                spans.append((opener_index, index + 2))
            index += 2
            continue

        char = value[index]
        if char in "([{":
            delimiter_stack.append((char, index))
        elif char in matching and delimiter_stack:
            opener, opener_index = delimiter_stack[-1]
            if opener == matching[char] or (
                opener in "(["
                and char in ")]"
                and "," in value[opener_index + 1 : index]
            ):
                delimiter_stack.pop()
                spans.append((opener_index, index + 1))
        index += 1
    return spans, bool(
        delimiter_stack
        or environment_stack
        or display_stack
        or display_invalid
    )


def _answer_block_structure_is_open(value: str) -> bool:
    """Return whether a multiline answer still has an open math wrapper."""

    _spans, is_open = _answer_structure_spans(value)
    return is_open


def _is_answer_display_fence_line(value: str) -> bool:
    return re.fullmatch(
        r"`{3,}(?:(?:latex|math)\s*)?",
        value.strip(),
        re.I,
    ) is not None


def _normalize_answer_block_line(
    line: str,
    *,
    presentation_prefix: bool = False,
) -> str:
    """Remove Markdown presentation prefixes from one semantic answer line.

    The returned text is used only for block structure decisions; scoring
    continues to use the original completion verbatim.  Iteration handles
    nested quote/list/task-list combinations such as ``> 1) - [x]``.  A
    leading emphasis token is presentation even when its matching closer sits
    after an answer cue and therefore is outside the supplied prefix slice.
    """

    value = line.strip()

    # Markdown may wrap either the field name (``**x**=``) or the entire
    # unfinished field (``**y=**``).  Remove only balanced markup adjacent to
    # assignment/component syntax; ordinary emphasis elsewhere remains prose.
    identifier = (
        r"(?:[A-Za-z]\w*|\\[A-Za-z]+)"
        r"(?:\s*[_^]\s*(?:\{[^{}]*\}|\w+))*"
    )
    value = re.sub(
        rf"(?P<mark>\*{{1,3}}|_{{1,3}}|`{{1,3}}|~~)"
        rf"(?P<body>\s*{identifier}\s*(?:&\s*)?(?:=|:)\s*)"
        rf"(?P=mark)",
        lambda match: match.group("body"),
        value,
    )
    value = re.sub(
        rf"(?P<mark>\*{{1,3}}|_{{1,3}}|`{{1,3}}|~~)"
        rf"(?P<body>\s*{identifier}\s*)"
        rf"(?P=mark)(?=\s*(?:&\s*)?(?:=|:|\bis\b))",
        lambda match: match.group("body"),
        value,
    )
    while value:
        previous = value
        value = re.sub(r"^(?:>\s*)+", "", value).lstrip()
        list_marker = (
            r"^(?:[-+*]|\d+[.)])(?:[ \t]+|$)"
            if presentation_prefix
            else r"^(?:[-+*]|\d+[.)])[ \t]+"
        )
        value = re.sub(list_marker, "", value).lstrip()
        value = re.sub(
            r"^\[(?:[ xX])\](?:[ \t]+|$)",
            "",
            value,
        ).lstrip()
        if _is_answer_display_fence_line(value):
            return value.casefold()
        value = re.sub(
            r"^(?:\*{1,3}|_{1,3}|`{1,3}|~~)(?=\S|$)",
            "",
            value,
        ).lstrip()
        if value == previous:
            break
    # LaTeX spacing, textual conjunctions, and alignment ampersands are
    # presentation-level syntax around named components.  Canonicalize them
    # here so continuation and completeness consume the exact same semantic
    # block representation.
    value = re.sub(
        r"(?is)\\(?:text|mathrm|operatorname)\s*\{\s*"
        r"(and|or|also|separately|independently|alternatively|otherwise)"
        r"\s*\}",
        r" \1 ",
        value,
    )
    value = _LATEX_SPACING_COMMAND_RE.sub("", value)
    value = re.sub(r"\s*&\s*(?==|:)", " ", value)
    # Also remove emphasis wrapped around the leading semantic word, as in
    # ``**independently** \\(\\boxed{3}\\)``.
    value = re.sub(
        r"(?<=\w)(?:\*{1,3}|_{1,3}|`{1,3}|~~)"
        r"(?=\s*(?:[,;:]|\\\(|\\\[|\\boxed|=))",
        "",
        value,
    )
    value = re.sub(
        r"(?:\*{1,3}|_{1,3}|`{1,3}|~~)\s*$",
        "",
        value,
    ).rstrip()
    return value


def _normalize_answer_block_payload(payload: str) -> str:
    """Return the sole semantic representation used for block decisions."""

    return "\n".join(
        _normalize_answer_block_line(line)
        for line in payload.splitlines()
    ).strip()


_ANSWER_ASSIGNMENT_FIELD_ATOM = (
    r"(?:[A-Za-z]\w*|\\[A-Za-z]+)"
    r"(?:\s*[_^]\s*(?:\{[^{}]*\}|\w+))*"
)
_ANSWER_ASSIGNMENT_FIELD = (
    rf"(?:component_[\w.-]+|"
    rf"(?:subproblem|part|component)\s+[A-Za-z0-9_.-]+|"
    rf"{_ANSWER_ASSIGNMENT_FIELD_ATOM})"
)
_ANSWER_ASSIGNMENT_TOKEN_RE = re.compile(
    rf"(?is)(?<![\w\\])(?P<label>{_ANSWER_ASSIGNMENT_FIELD})"
    r"\s*(?:&\s*)?(?P<operator>:=|=|:|\bis\b)"
)
_ORDERED_ANSWER_ITEM_VALUE = (
    r"(?:\d+|[ivxlcdmIVXLCDM\u2160-\u2188]+)"
)
_ORDERED_ANSWER_ITEM_MARKER = (
    rf"(?:[\(\uff08]\s*{_ORDERED_ANSWER_ITEM_VALUE}\s*[\)\uff09]|"
    rf"{_ORDERED_ANSWER_ITEM_VALUE}\s*[.):\uff0e\uff09\uff1a]|"
    # Circled, parenthesized, and dingbat numbers all have Numeric_Type=Digit
    # or Numeric_Type=Numeric.  Keep the source glyph here so the canonical id
    # routine can map every family to the same numbered schema.
    r"[\u2460-\u249b\u24f5-\u24fe\u2776-\u277f])"
)


_ANSWER_HTML_TAG_RE = re.compile(
    r"(?is)<!--.*?-->|"
    r"<\s*(?P<closing>/)?\s*"
    r"(?P<tag>[A-Za-z][A-Za-z0-9:.-]*)"
    r"(?P<attrs>(?:\s+(?:\"[^\"]*\"|'[^']*'|[^'\">])*)?)"
    r"\s*(?P<self_closing>/)?\s*>"
)
_ANSWER_HTML_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "tfoot",
    "thead",
    "tr",
    "ul",
}
_ANSWER_HTML_INLINE_TAGS = {
    "a",
    "abbr",
    "b",
    "bdi",
    "bdo",
    "big",
    "br",
    "button",
    "cite",
    "code",
    "data",
    "del",
    "dfn",
    "em",
    "font",
    "i",
    "ins",
    "kbd",
    "label",
    "mark",
    "q",
    "s",
    "samp",
    "small",
    "span",
    "strike",
    "strong",
    "sub",
    "sup",
    "time",
    "tt",
    "u",
    "var",
    "wbr",
}
_ANSWER_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff]")


def _normalize_answer_html_markup(
    value: str,
    *,
    preserve_field_boundaries: bool,
) -> str:
    """Strip HTML syntax without admitting tag attributes as answer text.

    HTML is presentation in generated answers, but block/cell elements carry
    semantic field boundaries.  Tokenising complete tags before assignment
    lexing prevents attributes such as ``data-index='2'`` from becoming fake
    assignments.  Entity decoding is deliberately shared with semantic-value
    checks: Unicode space entities and zero-width separators are empty, while
    semantic entities such as ``&pi;`` retain their decoded value.
    """

    tag_matches = list(_ANSWER_HTML_TAG_RE.finditer(value))
    opened_tags = {
        match.group("tag").casefold()
        for match in tag_matches
        if match.group("tag") is not None
        and not match.group("closing")
        and not match.group("self_closing")
    }
    closed_tags = {
        match.group("tag").casefold()
        for match in tag_matches
        if match.group("tag") is not None and match.group("closing")
    }
    paired_custom_tags = opened_tags & closed_tags

    def replace_tag(match: re.Match[str]) -> str:
        tag = match.group("tag")
        if tag is None:  # HTML comment
            return ""
        tag = tag.casefold()
        recognized = (
            tag in _ANSWER_HTML_BLOCK_TAGS
            or tag in _ANSWER_HTML_INLINE_TAGS
            or tag in paired_custom_tags
            or match.group("self_closing") is not None
            or match.group(0).rstrip().endswith("/>")
        )
        # A lone mathematical angle expression such as ``<x>`` is not HTML.
        # Unknown custom elements are presentation only when paired (or
        # explicitly self-closing), while ordinary HTML tags remain accepted.
        if not recognized:
            return match.group(0)
        if tag == "br":
            return "\n" if preserve_field_boundaries else " "
        if tag in {"td", "th"}:
            if preserve_field_boundaries and match.group("closing"):
                return " | "
            return ""
        if tag in _ANSWER_HTML_BLOCK_TAGS:
            if not preserve_field_boundaries:
                return " "
            # Container tags are presentation whitespace.  Table rows and
            # void rules alone create physical boundaries; table cells use a
            # visible-table delimiter above.  This avoids injecting either
            # punctuation or blank paragraphs inside a boxed value.
            if tag == "tr" and match.group("closing"):
                return "\n"
            if tag == "hr" or match.group(0).rstrip().endswith("/>"):
                return "\n"
            return " "
        return ""

    normalized = _ANSWER_HTML_TAG_RE.sub(replace_tag, value)
    normalized = html_lib.unescape(normalized)
    normalized = _ANSWER_ZERO_WIDTH_RE.sub("", normalized)
    return normalized


def _strip_assignment_markdown_markup(value: str) -> str:
    """Unwrap Markdown only when it encloses assignment-shaped content."""

    wrapped = re.compile(
        r"(?P<mark>\*{2,3}|_{1,3}|~~|`{1,3})"
        r"(?P<body>[^\r\n]{1,800}?)"
        r"(?P=mark)"
    )

    def unwrap(match: re.Match[str]) -> str:
        body = match.group("body")
        if re.search(
            rf"(?is)(?:^|\b){_ANSWER_ASSIGNMENT_FIELD_ATOM}\s*"
            r"(?:&\s*)?(?::=|=|:|\bis\b)",
            body,
        ) is not None:
            return body
        return match.group(0)

    previous = None
    while previous != value:
        previous = value
        value = wrapped.sub(unwrap, value)
    return value


def _ordered_answer_item_id(marker: str) -> str:
    match = re.search(
        r"(?i)(\d+|[ivxlcdm\u2160-\u2188]+|"
        r"[\u2460-\u249b\u24f5-\u24fe\u2776-\u277f])",
        marker,
    )
    if match is None:
        return "item"
    value = match.group(1).casefold()
    if value.isdecimal():
        return "".join(str(unicodedata.decimal(char)) for char in value)
    normalized = unicodedata.normalize("NFKC", value).casefold()
    if normalized.isdecimal():
        return "".join(str(unicodedata.decimal(char)) for char in normalized)
    numeric_value = unicodedata.numeric(value, None) if len(value) == 1 else None
    if numeric_value is not None and numeric_value.is_integer():
        return str(int(numeric_value))
    return normalized


def _normalize_authoritative_assignment_block(payload: str) -> str:
    """Canonicalize presentation variants before assignment lexing.

    This is intentionally one block-level pass.  HTML/Markdown line breaks,
    table fields, LaTeX environments and field wrappers, relation words, and
    ordered items all become the same assignment language.  Completeness and
    multi-box identity checks consume this representation rather than growing
    format-specific regular expressions at their call sites.
    """

    value = _normalize_answer_html_markup(
        payload,
        preserve_field_boundaries=True,
    )
    value = value.replace("⟨", r"\langle ").replace("⟩", r" \rangle")
    value = value.replace("〈", r"\langle ").replace("〉", r" \rangle")

    # Assignment labels are occasionally Markdown links (``[x=](#x)``).
    def unwrap_assignment_link(match: re.Match[str]) -> str:
        label = match.group("label")
        if re.fullmatch(
            rf"(?is)\s*{_ANSWER_ASSIGNMENT_FIELD_ATOM}\s*"
            r"(?:&\s*)?(?::=|=|:|\bis\b)\s*",
            label,
        ) is not None:
            return label
        return match.group(0)

    value = re.sub(
        r"\[(?P<label>[^\]\r\n]{1,160})\]\([^\)\r\n]*\)",
        unwrap_assignment_link,
        value,
    )
    value = _strip_assignment_markdown_markup(value)

    # Normalize field-name styling without unwrapping arbitrary mathematical
    # values.  The lookahead limits this to the left-hand side of a relation.
    value = re.sub(
        rf"(?is)\\(?:mathbf|vec|boldsymbol|overrightarrow|mathrm|mathit)"
        rf"\s*\{{\s*(?P<label>{_ANSWER_ASSIGNMENT_FIELD_ATOM})\s*\}}"
        r"(?=\s*(?:&\s*)?(?::=|=|:|\bis\b))",
        lambda match: match.group("label"),
        value,
    )
    value = re.sub(
        r"(?is)\\(?:text|textbf|mathrm|operatorname|mbox)\s*\{\s*"
        r"(and|or|also|separately|independently|alternatively|otherwise)"
        r"\s*\}",
        r" \1 ",
        value,
    )
    value = _LATEX_SPACING_COMMAND_RE.sub("", value)
    value = re.sub(r"\s*&\s*(?=:=|=|:|\bis\b)", " ", value)
    value = value.replace(":=", "=")

    # A LaTeX row separator is a semantic field boundary even when the whole
    # aligned environment is emitted on one physical line.
    value = re.sub(r"\\\\(?:\s*\[[^\]\r\n]*\])?", "\n", value)

    # Preserve ordered naked boxes by giving them a synthetic assignment name;
    # named items merely shed their presentation marker.  Empty ordered items
    # receive the same synthetic family and therefore invalidate a preceding
    # populated sibling.
    # A literal ``(`` may introduce an ordered item inside a larger wrapper,
    # but must not be stolen from a parenthesized option marker inside a LaTeX
    # content group (for example ``\\text{(D) }``).  At a true item boundary
    # the opening parenthesis is either part of the full ``(iv)`` marker or is
    # not immediately preceded by ``{``.
    # Ordered fields begin the authoritative block or follow a real field
    # boundary.  Commas and an arbitrary opening parenthesis are deliberately
    # excluded: treating them as boundaries turns tuple tails ``(2,3)`` and
    # function calls ``floor(3)`` into synthetic empty assignments.
    boundary = r"(?P<prefix>^|[\n;\[])(?P<space>\s*)"
    marker = rf"(?P<marker>{_ORDERED_ANSWER_ITEM_MARKER})\s*"

    def is_option_marker(match: re.Match[str]) -> bool:
        # Parenthesized A-D is an MCQ answer, not a Roman-numbered field.  The
        # glyph D happens to be a legal Roman numeral, so resolve this overlap
        # at the shared marker tokenizer instead of letting it create a stale
        # ``component_d`` assignment.
        normalized = unicodedata.normalize(
            "NFKC", match.group("marker")
        ).strip()
        return re.fullmatch(r"\(\s*[A-Da-d]\s*\)", normalized) is not None

    def synthetic_item(match: re.Match[str]) -> str:
        if is_option_marker(match):
            return match.group(0)
        item_id = _ordered_answer_item_id(match.group("marker"))
        return f"{match.group('prefix')} component_{item_id}="

    def strip_named_item_marker(match: re.Match[str]) -> str:
        if is_option_marker(match):
            return match.group(0)
        return f"{match.group('prefix')} "

    value = re.sub(
        rf"(?m){boundary}{marker}(?=\\boxed\s*\{{)",
        synthetic_item,
        value,
    )
    value = re.sub(
        rf"(?m){boundary}{marker}"
        rf"(?=(?:{_ANSWER_ASSIGNMENT_FIELD_ATOM})\s*"
        r"(?:&\s*)?(?::=|=|:|\bis\b))",
        strip_named_item_marker,
        value,
    )
    value = re.sub(
        rf"(?m){boundary}{marker}"
        r"(?=$|[\]\)}]|\\(?:end|right|rangle)\b|`{3,}|\$\$)",
        synthetic_item,
        value,
    )

    normalized_lines: list[str] = []
    for line in value.splitlines():
        normalized = _normalize_answer_block_line(line)
        if _is_answer_display_fence_line(normalized):
            continue
        normalized_lines.append(normalized)
    value = "\n".join(normalized_lines)
    # Markdown table bars delimit fields.  Escaped bars remain mathematical.
    value = re.sub(r"(?<!\\)\|", ";", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return value.strip()


def _canonical_answer_assignment_label(label: str) -> str:
    compact = re.sub(r"\s+", "", label).casefold()
    synthetic = re.fullmatch(r"component_(?P<item>[\w.-]+)", compact)
    if synthetic is not None:
        return f"number:{synthetic.group('item')}"
    group = re.fullmatch(
        r"(?P<kind>subproblem|part|component)(?P<item>[a-z0-9_.-]+)",
        compact,
    )
    if group is not None:
        return f"{group.group('kind')}:{group.group('item')}"
    return compact


_ANSWER_VALUE_CONTENT_WRAPPER_RE = re.compile(
    r"(?is)\\(?P<name>"
    r"text|textrm|textsf|texttt|textup|textnormal|textbf|textmd|textit|"
    r"textsl|textsc|emph|mbox|hbox|mathrm|mathbf|mathsf|mathtt|mathit|"
    r"mathnormal|operatorname|boldsymbol|underline|overline|smash|ensuremath|"
    r"fbox"
    r")\*?\s*\{(?P<content>[^{}]*)\}"
)
_ANSWER_VALUE_INVISIBLE_WRAPPER_RE = re.compile(
    r"(?is)\\(?:phantom|hphantom|vphantom)\s*\{[^{}]*\}"
)
_ANSWER_VALUE_STYLE_WRAPPER_RE = re.compile(
    r"(?is)\\(?:color|textcolor|colorbox)\*?\s*"
    r"(?:\[[^\]\r\n]*\]\s*)?\{[^{}]*\}\s*\{(?P<content>[^{}]*)\}"
)
_ANSWER_VALUE_COLOR_DECLARATION_RE = re.compile(
    r"(?is)\\(?:color|pagecolor)\*?\s*"
    r"(?:\[[^\]\r\n]*\]\s*)?\{[^{}]*\}"
)


def _semantic_answer_value_text(value: str) -> str:
    """Peel presentation-only wrappers while preserving semantic content.

    This is recursive over innermost balanced LaTeX wrappers.  A textual or
    styling wrapper contributes its contents; an invisible ``phantom`` family
    contributes nothing.  HTML presentation and comments follow the same
    rule.  Unknown mathematical commands remain semantic symbols.
    """

    semantic = _normalize_answer_html_markup(
        value,
        preserve_field_boundaries=False,
    )
    semantic = _LATEX_SPACING_COMMAND_RE.sub("", semantic)
    semantic = re.sub(
        r"(?is)\\(?:relax|displaystyle|textstyle|scriptstyle|scriptscriptstyle)\b",
        "",
        semantic,
    )
    previous = None
    while previous != semantic:
        previous = semantic
        semantic = _ANSWER_VALUE_INVISIBLE_WRAPPER_RE.sub("", semantic)
        semantic = _ANSWER_VALUE_STYLE_WRAPPER_RE.sub(
            lambda match: match.group("content"),
            semantic,
        )
        semantic = _ANSWER_VALUE_CONTENT_WRAPPER_RE.sub(
            lambda match: match.group("content"),
            semantic,
        )
    semantic = _ANSWER_VALUE_COLOR_DECLARATION_RE.sub("", semantic)
    return semantic


def _balanced_tex_group(
    value: str,
    start: int,
    *,
    opener: str,
    closer: str,
) -> tuple[str, int] | None:
    """Return one balanced TeX group and the cursor after its closer."""

    if start >= len(value) or value[start] != opener:
        return None
    depth = 1
    cursor = start + 1
    while cursor < len(value):
        char = value[cursor]
        escaped = cursor > 0 and value[cursor - 1] == "\\"
        if not escaped and char == opener:
            depth += 1
        elif not escaped and char == closer:
            depth -= 1
            if depth == 0:
                return value[start + 1 : cursor], cursor + 1
        cursor += 1
    return None


def _consume_tex_required_argument(
    value: str,
    start: int,
) -> tuple[str, int] | None:
    """Consume one mandatory TeX argument in grouped or single-token form."""

    cursor = start
    while cursor < len(value) and value[cursor].isspace():
        cursor += 1
    if cursor >= len(value) or value[cursor] == "}":
        return None
    if value[cursor] == "{":
        return _balanced_tex_group(
            value,
            cursor,
            opener="{",
            closer="}",
        )
    if value[cursor] == "\\":
        command = re.match(r"\\(?:[A-Za-z]+|.)", value[cursor:])
        if command is None:
            return None
        return command.group(0), cursor + command.end()
    return value[cursor], cursor + 1


def _tex_argument_has_semantic_value(value: str) -> bool:
    if not _latex_required_arguments_are_semantically_complete(value):
        return False
    semantic = _semantic_answer_value_text(value)
    semantic = _without_latex_spacing(semantic)
    semantic = semantic.strip().strip("$*_`;|&()[]{}")
    return bool(semantic and re.search(r"[\w\d\\]", semantic, re.UNICODE))


def _latex_required_arguments_are_semantically_complete(value: str) -> bool:
    r"""Validate mandatory arguments of structural TeX operators.

    Balanced braces alone do not make ``\frac{1}{}`` or ``\sqrt{}`` a
    value.  This scanner accepts both ordinary grouped arguments and TeX's
    legal single-token form while requiring every semantic slot to contain a
    real leaf after presentation wrappers, HTML, entities, and spacing are
    removed.
    """

    projected = _normalize_answer_html_markup(
        value,
        preserve_field_boundaries=False,
    )
    command_re = re.compile(r"\\(?P<name>dfrac|tfrac|frac|sqrt)\b")
    for command in command_re.finditer(projected):
        cursor = command.end()
        if command.group("name") == "sqrt":
            while cursor < len(projected) and projected[cursor].isspace():
                cursor += 1
            if cursor < len(projected) and projected[cursor] == "[":
                optional = _balanced_tex_group(
                    projected,
                    cursor,
                    opener="[",
                    closer="]",
                )
                if optional is None:
                    return False
                _degree, cursor = optional
            argument_count = 1
        else:
            argument_count = 2
        for _ in range(argument_count):
            argument = _consume_tex_required_argument(projected, cursor)
            if argument is None:
                return False
            content, cursor = argument
            if not _tex_argument_has_semantic_value(content):
                return False
    return True


def _assignment_rhs_has_complete_value(rhs: str) -> bool:
    """Return whether one canonical assignment has a concrete complete RHS."""

    if (
        not _latex_brace_groups_balanced(rhs)
        or not _latex_required_arguments_are_semantically_complete(rhs)
    ):
        return False

    if r"\boxed" in rhs:
        # Validate the recursive tree rather than comparing top-level spans
        # with every textual ``\\boxed`` mention.  A nested non-empty box is a
        # legitimate semantic leaf; any unclosed or empty descendant remains
        # fail-closed for an authoritative assignment.
        return _answer_boxes_are_semantically_complete(
            rhs,
            allow_terminal_unclosed=False,
        )

    value = _semantic_answer_value_text(rhs)
    value = re.sub(r"(?is)\\end\s*\{[^{}]+\}", " ", value)
    value = re.sub(
        r"(?is)\\(?:right\s*)?(?:rangle|[\]\)])|\\[\]\)]|\$\$|`{3,}",
        " ",
        value,
    )
    value = re.sub(
        r"(?is)\b(?:and|also|or|separately|independently|alternatively|otherwise)\b",
        " ",
        value,
    )
    value = value.strip().strip(",;|&()[]{}$*_`").strip()
    if not value or _TRAILING_INCOMPLETE_OPERATOR_RE.search(value):
        return False
    return bool(re.search(r"[\w\d\\]", value, re.UNICODE))


def _lex_authoritative_answer_assignments(payload: str) -> list[_AnswerAssignment]:
    """Lex all assignments from the canonical authoritative answer block."""

    normalized = _normalize_authoritative_assignment_block(payload)
    box_spans = _closed_box_spans_without_validation(normalized)
    tokens = [
        match
        for match in _ANSWER_ASSIGNMENT_TOKEN_RE.finditer(normalized)
        if not any(start <= match.start() < end for start, end in box_spans)
    ]
    assignments: list[_AnswerAssignment] = []
    for index, match in enumerate(tokens):
        rhs_end = tokens[index + 1].start() if index + 1 < len(tokens) else len(normalized)
        rhs = normalized[match.end() : rhs_end]
        label = _canonical_answer_assignment_label(match.group("label"))
        assignments.append(
            _AnswerAssignment(
                label=label,
                schema=_named_answer_component_schema(label),
                rhs=rhs,
                rhs_complete=_assignment_rhs_has_complete_value(rhs),
                start=match.start(),
                end=rhs_end,
            )
        )
    return assignments


def _answer_block_identity_payload(payload: str) -> str:
    """Remove display-only wrappers while retaining the enclosed answer."""

    value = _normalize_answer_block_payload(payload).strip()
    lines = value.splitlines()
    if lines and _is_answer_display_fence_line(lines[0]):
        lines = lines[1:]
        if lines and _is_answer_display_fence_line(lines[-1]):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    wrapper_pairs = ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"))
    changed = True
    while value and changed:
        changed = False
        for opener, closer in wrapper_pairs:
            if value.startswith(opener):
                value = value[len(opener) :].lstrip()
                if value.endswith(closer):
                    value = value[: -len(closer)].rstrip()
                changed = True
                break
    return value


def _answer_block_has_display_wrapper(payload: str) -> bool:
    normalized = _normalize_answer_block_payload(payload)
    return bool(
        any(_is_answer_display_fence_line(line) for line in normalized.splitlines())
        or re.search(r"\\[\[\]()]+|\$\$", normalized) is not None
    )


def _line_opens_answer_display_wrapper(line: str) -> bool:
    normalized = _normalize_answer_block_line(line)
    return normalized in {r"\[", r"\(", "$$"} or _is_answer_display_fence_line(
        normalized
    )


def _line_contains_only_boxed_answers(line: str) -> bool:
    line = _normalize_answer_block_line(line)
    spans: list[tuple[int, int]] = []
    marker = r"\boxed{"
    cursor = 0
    while True:
        start = line.find(marker, cursor)
        if start < 0:
            break
        depth = 1
        end = start + len(marker)
        while end < len(line) and depth:
            if line[end] == "{":
                depth += 1
            elif line[end] == "}":
                depth -= 1
            end += 1
        if depth:
            return False
        spans.append((start, end))
        cursor = end
    if not spans:
        return False
    residual = line
    for start, end in reversed(spans):
        residual = residual[:start] + residual[end:]
    for wrapper in (r"\(", r"\)", r"\[", r"\]", "$", r"\left", r"\right"):
        residual = residual.replace(wrapper, "")
    return not residual.strip(" \t,;.!*_`")


def _semantic_answer_continuation_payload(line: str) -> str | None:
    """Return the payload of a relation-led answer continuation line.

    Markdown quote/list markers and indentation are presentation only.  The
    relation grammar covers conjunction, disjunction, and alternative-value
    continuations; explanatory prose such as ``For example`` is deliberately
    outside it and therefore remains a section boundary.
    """

    value = _normalize_answer_block_line(line)
    relation = re.match(
        r"(?ix)^(?:"
        r"(?:and|or)(?:\s*,?\s*"
        r"(?:also|separately|independently|alternatively))?"
        r"|alternatively|separately|independently|otherwise"
        r"|as\s+(?:a|an)\s+alternative"
        r")\b\s*(?:[,;:]\s*)?",
        value,
    )
    if relation is None:
        return None
    return value[relation.end() :].strip()


def _line_is_semantic_answer_continuation(line: str) -> bool:
    payload = _semantic_answer_continuation_payload(line)
    if payload is None:
        return False
    return not payload or _explicit_payload_has_structured_math_lead(payload)


def _named_answer_component_label(
    prefix: str,
    *,
    allow_group_prefix: bool = False,
) -> str | None:
    """Return a canonical assignment label immediately before one box.

    A plain ``and`` is a presentation-level conjunction between components.
    Alternative/disjunctive relations deliberately do not qualify, even when
    each alternative happens to have a variable name.
    """

    value = _normalize_answer_block_line(prefix.rsplit("\n", 1)[-1])
    value = value.lstrip(" \t,;")
    if re.match(
        r"(?ix)^(?:"
        r"or|independently|separately|alternatively|otherwise|"
        r"as\s+(?:a|an)\s+alternative|"
        r"and\s*,?\s*(?:independently|separately|alternatively)"
        r")\b",
        value,
    ) is not None:
        return None
    value = re.sub(
        r"(?i)^(?:(?:and\s+)?also|and)\b(?:\s+|,\s*)",
        "",
        value,
    ).lstrip(" \t,;")
    # Parenthetical prose can document how one named component was computed;
    # it is not a relation between that component and the following one.
    previous = None
    while previous != value:
        previous = value
        value = re.sub(
            r"(?is)^\s*(?:"
            r"\([^()\\]*\)|\[[^\[\]\\]*\]|"
            r"\\(?:text|mathrm|operatorname)\s*\{[^{}]*\}"
            r")\s*[,;]?\s*",
            "",
            value,
        )
    value = re.sub(
        r"(?is)^(?:(?:\\left\s*)?(?:\\langle|\\[{}]|[([{])\s*)+",
        "",
        value,
    )
    group_prefix = (
        r"(?:(?:subproblem|part|component)\s+[A-Za-z0-9_.-]+\s*:\s*)?"
        if allow_group_prefix
        else ""
    )
    match = re.fullmatch(
        rf"(?is){group_prefix}(?P<label>(?:[A-Za-z]\w*|\\[A-Za-z]+)"
        r"(?:\s*[_^]\s*(?:\{[^{}]*\}|\w+))*)\s*=\s*"
        r"(?:(?:\\left\s*)?(?:\\langle|\\[{}]|[([{]))?\s*",
        value,
    )
    if match is None:
        match = re.fullmatch(
            rf"(?is){group_prefix}(?P<label>(?:[A-Za-z]\w*|\\[A-Za-z]+)"
            r"(?:\s*[_^]\s*(?:\{[^{}]*\}|\w+))*)\s+is\s*"
            r"(?:(?:\\left\s*)?(?:\\langle|\\[{}]|[([{]))?\s*",
            value,
        )
    if match is not None:
        return re.sub(r"\s+", "", match.group("label")).casefold()

    if allow_group_prefix:
        explicit_group = re.fullmatch(
            r"(?is)(?P<kind>subproblem|part|component)\s+"
            r"(?P<label>[A-Za-z0-9_.-]+)\s*(?::|=|\bis\b)\s*",
            value,
        )
        if explicit_group is not None:
            kind = explicit_group.group("kind").casefold()
            label = explicit_group.group("label").casefold()
            return f"{kind}:{label}"
        numbered = re.fullmatch(
            r"(?is)(?P<label>\d+)\s*(?::|[.)])\s*",
            value,
        )
        if numbered is not None:
            return f"number:{numbered.group('label')}"

    # A colon is a component separator only for a recognized component schema.
    # This admits ``x:``/``v_2:``/``width:`` without turning discourse labels
    # such as ``why:`` into answer components.
    colon = re.fullmatch(
        r"(?is)(?P<label>(?:[A-Za-z]\w*|\\[A-Za-z]+)"
        r"(?:\s*[_^]\s*(?:\{[^{}]*\}|\w+))*)\s*:\s*",
        value,
    )
    if colon is not None:
        label = re.sub(r"\s+", "", colon.group("label")).casefold()
        if _named_answer_component_schema(label) is not None:
            return label
    return None


def _named_answer_component_schema(label: str) -> str | None:
    """Return a conservative family for a continuous component sequence.

    The family is intentionally narrower than the complete-component parser:
    arbitrary identifiers can still form complete named answers, but a bare
    later ``why=`` cannot extend an ``x=...`` block.  The admitted continuation
    families cover coordinate/scalar symbols, indexed vector components,
    explicit ``component/part/subproblem N`` labels, and numbered fields.
    """

    compact = re.sub(r"\s+", "", label).casefold()
    group = re.fullmatch(
        r"(?P<kind>subproblem|part|component):[a-z0-9_.-]+",
        compact,
    )
    if group is not None:
        return f"group:{group.group('kind')}"
    if re.fullmatch(r"number:[\w.-]+", compact) is not None:
        return "numbered"
    if re.fullmatch(r"(?:\\[a-z]+|[a-z])", compact) is not None:
        return "symbolic"
    indexed = re.fullmatch(
        r"(?P<base>\\[a-z]+|[a-z])"
        r"(?:\d+|(?:[_^](?:\{[^{}]+\}|[a-z0-9]+)))+",
        compact,
    )
    if indexed is not None:
        return f"indexed:{indexed.group('base')}"
    if re.fullmatch(r"[a-z][a-z0-9_]*", compact) is not None and compact not in {
        "analysis",
        "answer",
        "because",
        "calculation",
        "caveat",
        "check",
        "comment",
        "commentary",
        "conclusion",
        "derivation",
        "example",
        "explanation",
        "hence",
        "how",
        "note",
        "notes",
        "proof",
        "question",
        "rationale",
        "reason",
        "reasoning",
        "response",
        "result",
        "solution",
        "summary",
        "therefore",
        "thus",
        "verification",
        "value",
        "verify",
        "why",
        "work",
        "working",
    }:
        return "named"
    return None


_ANSWER_BOX_OPEN_RE = re.compile(r"\\boxed\s*\{")
_ANSWER_BOX_MENTION_RE = re.compile(r"\\boxed\b")


def _answer_box_forest(value: str) -> tuple[_AnswerBoxNode, ...]:
    """Parse every ``\\boxed`` wrapper into a recursive containment forest.

    Each opening is balanced independently so malformed ancestors remain
    visible while a closed descendant can still be described.  Tree roots are
    the only independent answer identities; descendants are presentation of a
    root's semantic value.
    """

    raw_nodes: list[tuple[int, int, int, int | None]] = []
    for match in _ANSWER_BOX_OPEN_RE.finditer(value):
        depth = 1
        index = match.end()
        while index < len(value) and depth:
            if value[index] == "{":
                depth += 1
            elif value[index] == "}":
                depth -= 1
            index += 1
        end = index if depth == 0 else None
        content_end = end - 1 if end is not None else len(value)
        raw_nodes.append((match.start(), match.end(), content_end, end))

    def build(index: int) -> tuple[_AnswerBoxNode, int]:
        start, content_start, content_end, end = raw_nodes[index]
        limit = end if end is not None else len(value) + 1
        children: list[_AnswerBoxNode] = []
        cursor = index + 1
        while cursor < len(raw_nodes) and raw_nodes[cursor][0] < limit:
            child, cursor = build(cursor)
            children.append(child)
        return (
            _AnswerBoxNode(
                start=start,
                content_start=content_start,
                content_end=content_end,
                end=end,
                children=tuple(children),
            ),
            cursor,
        )

    roots: list[_AnswerBoxNode] = []
    cursor = 0
    while cursor < len(raw_nodes):
        root, cursor = build(cursor)
        roots.append(root)
    return tuple(roots)


def _iter_answer_box_nodes(
    nodes: Iterable[_AnswerBoxNode],
) -> Iterable[_AnswerBoxNode]:
    for node in nodes:
        yield node
        yield from _iter_answer_box_nodes(node.children)


def _answer_box_node_has_conflicting_alternatives(
    value: str,
    node: _AnswerBoxNode,
) -> bool:
    r"""Return whether one box contains disagreeing boxed alternatives.

    Nested boxes are normally presentation wrappers around one semantic value.
    They cease to be one value when distinct sibling payloads are joined by an
    explicit disjunction.  Inspecting the tree (rather than top-level regex
    spans) catches ``\boxed{\boxed{3}\text{ or }\boxed{4}}`` without
    rejecting ordinary tuple/vector children joined by commas or conjunctions.
    """

    content = value[node.content_start : node.content_end]
    if _answer_value_has_explicit_alternatives(content):
        return True
    if len(node.children) < 2:
        return False
    for previous, current in zip(node.children, node.children[1:]):
        previous_content = value[
            previous.content_start : previous.content_end
        ]
        current_content = value[current.content_start : current.content_end]
        previous_identity = _serialized_answer_identity(
            _semantic_answer_value_text(previous_content)
        )
        current_identity = _serialized_answer_identity(
            _semantic_answer_value_text(current_content)
        )
        if not previous_identity or not current_identity:
            continue
        bridge_start = previous.end
        if bridge_start is None:
            continue
        bridge = _semantic_answer_value_text(
            value[bridge_start : current.start]
        )
        has_alternative_relation = re.search(
            r"(?ix)(?:"
            r"\b(?:or|alternatively|otherwise|alternatives?)\b|"
            r"\\(?:lor|vee)\b"
            r")",
            bridge,
        ) is not None
        if has_alternative_relation and previous_identity != current_identity:
            return True
    return False


def _answer_value_has_explicit_alternatives(value: str) -> bool:
    r"""Detect distinct visible values joined by an explicit prose choice.

    This intentionally recognizes prose relations (``or``, ``alternatively``
    and ``otherwise``), including relations carried by ``\text{...}``.  It
    does not reinterpret the mathematical operator ``\lor`` or substrings in
    names such as ``floor``.  Each side must retain semantic answer content,
    and identical alternatives remain one identity.
    """

    visible = _semantic_answer_value_text(value)
    relation_re = re.compile(
        r"(?i)\b(?:or|alternatively|otherwise|alternatives?)\b"
    )
    for relation in relation_re.finditer(visible):
        left = visible[: relation.start()]
        right = visible[relation.end() :]
        left = re.split(r"[;\n]", left)[-1]
        right = re.split(r"[;\n]", right)[0]
        left = re.sub(r"(?i)^\s*(?:either|one\s+of)\b", "", left)
        left = left.strip(" \t\r\n,;:|&()[]{}$*_`")
        right = right.strip(" \t\r\n,;:|&()[]{}$*_`")
        if not left or not right:
            continue
        if not re.search(r"[\w\d\\]", left, re.UNICODE) or not re.search(
            r"[\w\d\\]", right, re.UNICODE
        ):
            continue
        left_identity = _serialized_answer_identity(left)
        right_identity = _serialized_answer_identity(right)
        if left_identity and right_identity and left_identity != right_identity:
            return True
    return False


def _answer_box_tree_has_conflicting_alternatives(value: str) -> bool:
    return any(
        _answer_box_node_has_conflicting_alternatives(value, node)
        for node in _iter_answer_box_nodes(_answer_box_forest(value))
    )


def _answer_box_node_has_complete_semantic_value(
    value: str,
    node: _AnswerBoxNode,
    *,
    allow_unclosed: bool,
) -> bool:
    """Validate one box and every semantic leaf below it."""

    if node.end is None and not allow_unclosed:
        return False
    if any(
        not _answer_box_node_has_complete_semantic_value(
            value,
            child,
            allow_unclosed=False,
        )
        for child in node.children
    ):
        return False
    if _answer_box_node_has_conflicting_alternatives(value, node):
        return False

    content = value[node.content_start : node.content_end]
    if not node.children and _EMPTY_OR_PLACEHOLDER_BOX_RE.fullmatch(
        rf"\boxed{{{content}}}"
    ) is not None:
        return False

    # Replace each already-validated child by a scalar sentinel.  This retains
    # the parent's operators and wrappers, so ``child +`` stays incomplete
    # while ``3 + child`` remains a complete expression.
    rendered = content
    for child in reversed(node.children):
        child_end = child.end
        if child_end is None:
            return False
        relative_start = child.start - node.content_start
        relative_end = child_end - node.content_start
        rendered = rendered[:relative_start] + "1" + rendered[relative_end:]
    return _answer_candidate_is_complete(rendered)


def _answer_boxes_are_semantically_complete(
    value: str,
    *,
    allow_terminal_unclosed: bool,
) -> bool:
    """Validate all box roots, their balance, and their recursive leaves."""

    roots = _answer_box_forest(value)
    nodes = tuple(_iter_answer_box_nodes(roots))
    if not roots or len(nodes) != len(_ANSWER_BOX_MENTION_RE.findall(value)):
        return False

    unclosed = [node for node in nodes if node.end is None]
    if unclosed:
        # Candidate extraction preserves the established final-stage contract:
        # one terminal pre-opened root may contribute its complete content.
        # Authoritative assignment RHS validation never permits an open box.
        if (
            not allow_terminal_unclosed
            or len(unclosed) != 1
            or unclosed[0] not in roots
        ):
            return False
    return all(
        _answer_box_node_has_complete_semantic_value(
            value,
            root,
            allow_unclosed=allow_terminal_unclosed and root.end is None,
        )
        for root in roots
    )


def _closed_box_spans_without_validation(value: str) -> list[tuple[int, int]]:
    """Locate balanced top-level ``\\boxed`` spans without completeness."""

    return [
        (node.start, node.end)
        for node in _answer_box_forest(value)
        if node.end is not None
    ]


def _answer_block_named_component_labels(payload: str) -> list[str]:
    """Collect complete component labels through the shared assignment lexer."""

    return [
        assignment.label
        for assignment in _lex_authoritative_answer_assignments(payload)
        if assignment.schema is not None and assignment.rhs_complete
    ]


def _answer_block_has_incomplete_named_component_tail(payload: str) -> bool:
    """Detect a missing final value in an established component schema.

    Both block continuation and completeness call this function on the same
    normalized payload.  Only a *continuous* final line whose label belongs to
    a previously established component family qualifies.  Consequently
    ``x=\\boxed{2}\ny=`` is incomplete, while a following prose line
    ``why=`` remains outside the answer block.
    """

    assignments = _lex_authoritative_answer_assignments(payload)
    schemas: dict[str, list[_AnswerAssignment]] = {}
    for assignment in assignments:
        if assignment.schema is not None:
            schemas.setdefault(assignment.schema, []).append(assignment)
    if any(
        len(members) > 1
        and any(member.rhs_complete for member in members)
        and any(not member.rhs_complete for member in members)
        for members in schemas.values()
    ):
        return True

    # ``_normalize_answer_block_payload`` predates the authoritative lexer and
    # intentionally strips the first list marker on a physical line.  Retain a
    # fail-closed bridge for callers that pass that already-normalized form:
    # a surviving empty numbered item after any complete box is its sibling.
    return bool(
        _closed_box_spans_without_validation(payload)
        and any(
            assignment.schema == "numbered" and not assignment.rhs_complete
            for assignment in assignments
        )
    )


def _line_is_grouped_answer_component_continuation(
    accumulated: str,
    line: str,
) -> bool:
    """Admit only continuous explicit group/number component schemas."""

    normalized = _normalize_answer_block_line(line)
    box_start = normalized.find(r"\boxed{")
    if box_start < 0:
        return False
    label = _named_answer_component_label(
        normalized[:box_start],
        allow_group_prefix=True,
    )
    if label is None:
        return False
    schema = _named_answer_component_schema(label)
    if schema is None or not (
        schema.startswith("group:") or schema == "numbered"
    ):
        return False
    return any(
        _named_answer_component_schema(prior_label) == schema
        for prior_label in _answer_block_named_component_labels(accumulated)
    )


def _line_is_named_answer_component(
    line: str,
    *,
    allow_group_prefix: bool = False,
) -> bool:
    normalized = _normalize_answer_block_line(line)
    box_start = normalized.find(r"\boxed{")
    return box_start >= 0 and _named_answer_component_label(
        normalized[:box_start],
        allow_group_prefix=allow_group_prefix,
    ) is not None


def _line_is_answer_table_presentation(line: str) -> bool:
    """Recognize a two-column Markdown answer table header or divider."""

    value = _normalize_answer_block_line(line).strip()
    if re.fullmatch(
        r"(?is)\|?\s*(?:field|component|variable|name)\s*\|\s*"
        r"(?:value|answer|result)\s*\|?",
        value,
    ) is not None:
        return True
    return re.fullmatch(
        r"\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*",
        value,
    ) is not None


def _answer_block_uses_markdown_table_layout(payload: str) -> bool:
    """Return whether an accumulated answer block established table layout."""

    return any(
        _line_is_answer_table_presentation(line) for line in payload.splitlines()
    )


def _line_starts_explicit_answer_block(line: str) -> bool:
    """Accept answer-shaped content, not arbitrary post-heading prose."""

    normalized = _normalize_answer_block_line(line)
    return bool(
        normalized
        and (
            _line_opens_answer_display_wrapper(normalized)
            or _line_is_answer_table_presentation(normalized)
            or _line_contains_only_boxed_answers(normalized)
            or _line_is_semantic_answer_continuation(normalized)
            or _line_is_named_answer_component(
                normalized,
                allow_group_prefix=True,
            )
            or _explicit_payload_has_structured_math_lead(normalized)
        )
    )


def _line_continues_explicit_answer_block(
    accumulated: str,
    current_line: str,
    next_line: str,
) -> bool:
    """Recognize structural continuations without admitting ordinary prose."""

    if _answer_block_structure_is_open(accumulated):
        return True
    normalized_current = _normalize_answer_block_line(current_line)
    stripped_next = _normalize_answer_block_line(next_line)
    if not stripped_next:
        return False
    if _line_is_answer_table_presentation(stripped_next):
        return True
    if (
        _answer_block_uses_markdown_table_layout(accumulated)
        and "|" in stripped_next
        and any(
            assignment.schema is not None
            for assignment in _lex_authoritative_answer_assignments(stripped_next)
        )
    ):
        return True
    if _line_contains_only_boxed_answers(next_line):
        # Consecutive naked boxed lines must stay in one authority block so a
        # later conflict check can reject distinct final identities.
        return True
    if _line_is_semantic_answer_continuation(next_line):
        return True
    if _line_is_named_answer_component(next_line):
        return True
    if _line_is_grouped_answer_component_continuation(
        accumulated,
        next_line,
    ):
        return True
    if _answer_block_has_incomplete_named_component_tail(
        f"{accumulated}\n{stripped_next}"
    ):
        # Once a complete named component establishes the schema, a directly
        # adjacent missing sibling remains inside the same authority block.
        # Completeness then invalidates the whole block instead of allowing an
        # earlier boxed component to escape.
        return True
    if re.match(r"^(?:\\(?:begin|end)\b|[()\[\]{}]|\\[{}])", stripped_next):
        return True
    return normalized_current.rstrip().endswith((",", "&", r"\\", "=")) and bool(
        _CONCRETE_MATH_ANSWER_ANYWHERE_RE.search(stripped_next)
    )


def _explicit_answer_block_end(
    text: str,
    cue_end: int,
    *,
    max_chars: int = 8000,
) -> int:
    """Return the bounded end of a physical-line explicit answer block.

    A block continues while a tuple/set/interval/matrix wrapper is open, or
    across consecutive boxed-only answer lines.  It stops at the next blank
    paragraph, heading, answer cue, closing think tag, or ordinary prose once
    the math structure is closed.
    """

    limit = min(len(text), cue_end + max_chars)
    cursor = cue_end
    last_content_end = cue_end
    accumulated = ""
    started = False
    while cursor < limit:
        line_end = cursor
        while line_end < limit and text[line_end] not in "\r\n":
            line_end += 1
        line = text[cursor:line_end]
        think_end = line.find("</think>")
        if think_end >= 0:
            line_end = cursor + think_end
            line = text[cursor:line_end]
        stripped = _normalize_answer_block_line(line)
        if not stripped:
            if started:
                break
        else:
            if (
                re.match(r"^\s{0,3}#{1,6}\s+\S", line)
                or _EXPLICIT_ANSWER_CUE_RE.match(line) is not None
            ):
                break
            if not started and not _line_starts_explicit_answer_block(line):
                break
            started = True
            accumulated = (
                f"{accumulated}\n{stripped}" if accumulated else stripped
            )
            last_content_end = line_end
        if think_end >= 0 or line_end >= limit:
            break

        next_start = line_end + 1
        if text[line_end] == "\r" and next_start < limit and text[next_start] == "\n":
            next_start += 1
        next_end = next_start
        while next_end < limit and text[next_end] not in "\r\n":
            next_end += 1
        next_line = text[next_start:next_end]
        if started and not _line_continues_explicit_answer_block(
            accumulated,
            line,
            next_line,
        ):
            break
        cursor = next_start
    return last_content_end


def _explicit_answer_cue_span(
    text: str,
    *,
    cue_start: int,
    cue_end: int,
) -> tuple[int, bool]:
    """Return one cue's semantic end and whether it uses block layout."""

    uses_block_layout = _explicit_answer_uses_block_layout(
        text,
        cue_start=cue_start,
        cue_end=cue_end,
    )
    if uses_block_layout:
        return _explicit_answer_block_end(text, cue_end), True
    return _candidate_clause_end(text, cue_end), False


def _boxes_share_compound_answer_structure(
    text: str,
    boxes: list[_AnswerCandidate],
) -> bool:
    """Return whether multiple boxes belong to one tuple/set/matrix value."""

    first_start = boxes[0].start
    last_end = boxes[-1].end
    spans, _is_open = _answer_structure_spans(text)
    return any(start < first_start and end >= last_end for start, end in spans)


def _boxes_are_named_compound_components(
    payload: str,
    boxes: list[_AnswerCandidate],
) -> bool:
    """Return whether distinct assignments account for every boxed value."""

    assignments = _lex_authoritative_answer_assignments(payload)
    boxed_assignments = [
        assignment
        for assignment in assignments
        if assignment.schema is not None
        and assignment.rhs_complete
        and len(_closed_box_spans_without_validation(assignment.rhs)) == 1
    ]
    if len(boxed_assignments) != len(boxes):
        return False
    labels = [assignment.label for assignment in boxed_assignments]
    return len(labels) > 1 and len(set(labels)) == len(labels)


def _boxes_have_independent_relation(
    payload: str,
    boxes: list[_AnswerCandidate],
) -> bool:
    """Detect semantic alternatives between adjacent boxed identities."""

    for previous, current in zip(boxes, boxes[1:]):
        bridge = payload[previous.end : current.start]
        normalized = " ".join(
            filter(
                None,
                (
                    _normalize_answer_block_line(line)
                    for line in bridge.splitlines()
                ),
            )
        )
        structure_spans, _is_open = _answer_structure_spans(normalized)
        for relation in re.finditer(
            r"(?i)\b(?:or|independently|separately|alternatively)\b",
            normalized,
        ):
            if not any(
                start <= relation.start() < end
                for start, end in structure_spans
            ):
                return True
    return False


def _multi_box_identities_conflict(
    payload: str,
    boxes: list[_AnswerCandidate],
) -> bool:
    """Reject distinct independent identities while preserving compounds."""

    if len(boxes) < 2:
        return False
    identities = {_serialized_answer_identity(box.content) for box in boxes}
    if len(identities) <= 1:
        return False
    if _boxes_share_compound_answer_structure(payload, boxes):
        return False
    if _boxes_have_independent_relation(payload, boxes):
        return True
    if _boxes_are_named_compound_components(payload, boxes):
        return False
    # Distinct boxes are unsafe unless one compound structure accounts for
    # every identity.  This also prevents component boxes from escaping when
    # a surrounding explicit expression is truncated or otherwise invalid.
    return True


def _explicit_payload_has_structured_math_lead(payload: str) -> bool:
    """Reject prose examples while allowing common compound answer prefixes."""

    if any(
        assignment.schema is not None and assignment.rhs_complete
        for assignment in _lex_authoritative_answer_assignments(payload)
    ):
        return True
    value = _answer_block_identity_payload(payload).lstrip(" \t\r\n$`*_")
    first_line = value.splitlines()[0] if value else ""
    value_boxes = _boxed_answer_candidates(first_line)
    if value_boxes and _named_answer_component_label(
        first_line[: value_boxes[0].start],
        allow_group_prefix=True,
    ) is not None:
        return True
    if len(value_boxes) > 1 and _boxes_are_named_compound_components(
        first_line,
        value_boxes,
    ):
        return True
    if _CONCRETE_MATH_ANSWER_PREFIX_RE.match(value) is not None:
        return True
    if re.match(
        r"(?is)^(?:\\left\s*)?(?:\\langle\b|\\[{}]|[([{])",
        value,
    ) is not None:
        return True
    if re.match(
        r"(?is)^(?:(?:the\s+)?(?:value|result|solution)\s*(?:is|=)\s*)"
        r"(?:\\boxed\b|\\begin\b|(?:\\left\s*)?"
        r"(?:\\langle\b|\\[{}]|[([{]))",
        value,
    ) is not None:
        return True
    box_start = value.find(r"\boxed{")
    if box_start < 0:
        return False
    assignment_prefix = value[:box_start].strip()
    return re.fullmatch(
        r"(?is)[A-Za-z\\][A-Za-z0-9_\\,\s^{}]*\s*=\s*"
        r"(?:(?:\\left\s*)?(?:\\langle|\\[{}]|[([{]))?",
        assignment_prefix,
    ) is not None


def _answer_cue_is_contextual_meta(
    text: str, cue_start: int, payload: str
) -> bool:
    """Identify quoted, conditional, or format-only mentions of an answer."""

    line_prefix = text[text.rfind("\n", 0, cue_start) + 1 : cue_start]
    single_quotes = re.findall(r"(?<!\w)'|'(?!\w)", line_prefix)
    if len(single_quotes) % 2:
        return True
    if line_prefix.count('"') % 2 or line_prefix.count("“") > line_prefix.count("”"):
        return True
    if _CONDITIONAL_ANSWER_PREFIX_RE.search(text[max(0, cue_start - 48) : cue_start]):
        return True
    if _CONDITIONAL_ANSWER_SUFFIX_RE.search(payload):
        return True
    if _NON_ANSWER_PAYLOAD_RE.fullmatch(payload) is not None:
        return True
    if _NON_FINAL_ANSWER_CLAUSE_RE.search(payload):
        return True
    if _EMPTY_OR_PLACEHOLDER_BOX_RE.search(payload):
        return True
    return False


def _contextual_answer_cue_supersedes_prior(
    text: str, cue_start: int, payload: str
) -> bool:
    """Return whether later uncertain/meta answer text invalidates an old value.

    Quoted or conditional mentions merely discuss another answer and therefore
    remain ignorable.  A standalone later cue containing a concrete value is
    different: even if phrased as formatting or uncertainty (for example,
    ``the answer should be presented as approximately 50.7``), it is the
    model's latest semantic answer state.  Recovery must retain the raw text
    rather than resurrecting an older complete candidate.
    """

    line_prefix = text[text.rfind("\n", 0, cue_start) + 1 : cue_start]
    single_quotes = re.findall(r"(?<!\w)'|'(?!\w)", line_prefix)
    if len(single_quotes) % 2:
        return False
    if line_prefix.count('"') % 2 or line_prefix.count("“") > line_prefix.count("”"):
        return False
    if _CONDITIONAL_ANSWER_PREFIX_RE.search(text[max(0, cue_start - 48) : cue_start]):
        return False
    if _CONDITIONAL_ANSWER_SUFFIX_RE.search(payload) and not (
        _ANSWER_PRESENTATION_PREFIX_RE.search(payload)
    ):
        return False
    return _CONCRETE_MATH_ANSWER_ANYWHERE_RE.search(payload) is not None


def _answer_cue_payload_is_concrete(
    text: str,
    cue_start: int,
    cue_end: int,
    end: int,
    *,
    normalized_payload: str | None = None,
) -> bool:
    """Accept only an explicit, locally parseable answer literal.

    Answer-like prose in a restatement (``the answer should be in the form``),
    a quoted prior subproblem, or a conditional hypothesis must not become a
    recovery target.  Non-numeric free text is intentionally left to its
    existing evaluator; this recovery path is for structural math/MCQ values.
    """

    payload = (
        normalized_payload
        if normalized_payload is not None
        else text[cue_end:end].strip(" \t\r\n:=`*_")
    )
    if not payload or _answer_cue_is_contextual_meta(text, cue_start, payload):
        return False
    comparable_payload = _answer_block_identity_payload(payload).lstrip("$`*_ ")
    if comparable_payload.startswith(r"\$"):
        comparable_payload = comparable_payload[2:].lstrip()
    if comparable_payload.startswith(("\\(", "\\[")):
        comparable_payload = comparable_payload[2:].lstrip()
    return _CONCRETE_MATH_ANSWER_PREFIX_RE.match(comparable_payload) is not None


def _answer_cue_is_strong_boundary(text: str, cue_start: int, cue_text: str) -> bool:
    """Return whether a value-free cue still supersedes earlier candidates."""

    if cue_text.casefold().lstrip().startswith("final answer"):
        return True
    line_prefix = text[text.rfind("\n", 0, cue_start) + 1 : cue_start]
    return not _normalize_answer_block_line(
        line_prefix,
        presentation_prefix=True,
    )


def _answer_cue_strength(text: str, cue_start: int, cue_text: str) -> int:
    """Rank explicit presentation boundaries without inspecting a reference.

    A line-level ``Answer:``/``Final answer`` is stronger evidence than an
    incidental later ``the answer is`` inside an explanatory sentence. A
    clearly marked correction receives the same strength, allowing a genuine
    later revision to supersede an earlier presentation.
    """

    normalized_cue = cue_text.casefold().lstrip()
    if normalized_cue.startswith("final answer"):
        return 4
    if _ANSWER_REPLACEMENT_CUE_RE.match(cue_text):
        return 5
    if _ANSWER_COMMITMENT_CUE_RE.match(cue_text):
        return 4
    line_prefix = text[text.rfind("\n", 0, cue_start) + 1 : cue_start]
    if not _normalize_answer_block_line(
        line_prefix,
        presentation_prefix=True,
    ):
        return 4
    if _ANSWER_CORRECTION_PREFIX_RE.search(text[max(0, cue_start - 96) : cue_start]):
        return 4
    return 2


def _minimal_explicit_scalar(payload: str) -> str | None:
    """Return a terminal/concessive scalar while preserving signs and units.

    The accepted suffix is deliberately narrow: presentation punctuation, or
    a comma introducing an explanatory/concessive clause. Ordered tuples,
    lists, arithmetic continuations, and corrections therefore remain intact
    for the normal symbolic parser.
    """

    match = _EXPLICIT_SCALAR_PREFIX_RE.match(payload)
    if match is None:
        return None
    suffix = match.group("suffix").strip()
    value = match.group("value").strip()
    if not suffix and value.startswith(r"\$"):
        return value
    if value.startswith(("$", r"\$")) and not suffix.strip(".!ã€‚ï¼"):
        return value
    if _EXPLANATORY_SCALAR_SUFFIX_RE.match(suffix) is None:
        return None
    return value


def _answer_candidate_is_terminal(text: str, candidate: _AnswerCandidate) -> bool:
    """Return whether only presentational closers follow a candidate."""

    suffix = text[candidate.end :].strip()
    while suffix.lower().startswith("</think>"):
        suffix = suffix[len("</think>") :].strip()
    if _TERMINAL_CONCLUSION_CLOSER_RE.fullmatch(suffix) is not None:
        return True
    suffix = suffix.strip(" \t\r\n.!?\u3002\uff01\uff1f*_`$")
    return not suffix or suffix in {r"\]", r"\)"}


def _terminal_complete_result_candidate(text: str) -> _AnswerCandidate | None:
    """Return only the completed result sentence adjacent to a truncated tail.

    Long chain-of-thought often contains thousands of valid intermediate
    equalities.  Searching all of them would silently turn an early derivation
    into a final answer.  The one safe non-labelled recovery is the last
    completed sentence immediately before the unfinished suffix, and only
    when that sentence explicitly maps a result verb to a scalar value.
    """

    stripped = text.rstrip()
    sentence_ends = list(re.finditer(r"[.!。！？](?=\s|$)", stripped))
    if not sentence_ends:
        return None
    end = sentence_ends[-1].end()
    previous_end = sentence_ends[-2].end() if len(sentence_ends) > 1 else 0
    newline_start = stripped.rfind("\n", previous_end, end) + 1
    start = max(previous_end, newline_start)
    clause = stripped[start:end].strip()
    if (
        not _answer_candidate_is_complete(clause)
        or "?" in clause
        or "？" in clause
        or re.search(
            r"(?i)^\s*(?:suppose|assuming|if\b|for\s+example|e\.g\.)", clause
        )
    ):
        return None
    matches = list(_CONCLUSIVE_RESULT_VERB_RE.finditer(clause))
    if not matches:
        return None
    rhs = clause[matches[-1].end() :].strip().rstrip(".!。！？").strip()
    if _CONCLUSIVE_SCALAR_RHS_RE.match(rhs) is None:
        return None
    return _AnswerCandidate(start, end, clause, rhs, 1)


def _explicit_answer_candidates(text: str) -> list[_AnswerCandidate]:
    candidates = _boxed_answer_candidates(text)
    suppressed_component_box_starts: set[int] = set()
    for match in _CONCLUSIVE_OPTION_LABEL_RE.finditer(text):
        label = match.group(1).upper()
        candidates.append(
            _AnswerCandidate(
                match.start(),
                match.end(),
                f"Final answer: {label}",
                label,
                3,
            )
        )
    latest_semantic_cue_start = -1
    latest_semantic_cue_strength = -1
    semantic_boundary_blocks_recovery = False
    for match in _EXPLICIT_ANSWER_CUE_RE.finditer(text):
        end, _uses_block_layout = _explicit_answer_cue_span(
            text,
            cue_start=match.start(),
            cue_end=match.end(),
        )
        clause = text[match.start() : end].strip()
        # Strip presentation only.  ``=``/``:`` at the *end* are semantic
        # incompleteness markers (for example ``x=...\ny=``) and must survive
        # into the one normalized block representation.
        payload = text[match.end() : end].strip(" \t\r\n")
        normalized_payload = _normalize_answer_block_payload(payload)
        identity_payload = _answer_block_identity_payload(normalized_payload)
        minimal_scalar = _minimal_explicit_scalar(identity_payload)
        # A presentation-wrapped option label is a complete structured scalar
        # even when it is not numeric.  Canonicalizing it here gives boxed and
        # unboxed ``\text{(D)}`` the same candidate path; the MCQ verifier still
        # proves the question schema before treating the letter as a choice.
        structured_option_label = _reference_option_label(identity_payload)
        if minimal_scalar is None and structured_option_label is not None:
            minimal_scalar = structured_option_label
        cue_strength = _answer_cue_strength(
            text, match.start(), match.group()
        )
        is_contextual_meta = _answer_cue_is_contextual_meta(
            text, match.start(), identity_payload
        )
        source_stance = _source_evidence_stance(
            text,
            start=match.start(),
            end=end,
        )
        if is_contextual_meta and source_stance != "adopted":
            # A standalone later answer-like statement with a numeric payload
            # is still an authoritative boundary even when its wording is
            # conditional/format-oriented.  Do not select it, but do prevent
            # an older value from silently replacing it; raw parsing then
            # resolves the full later statement fail-closed.
            if _contextual_answer_cue_supersedes_prior(
                text, match.start(), identity_payload
            ):
                latest_semantic_cue_start = match.start()
                latest_semantic_cue_strength = 100
                semantic_boundary_blocks_recovery = True
            continue
        currency_scalar = bool(
            minimal_scalar
            and minimal_scalar.lstrip().startswith(("$", r"\$"))
        )
        authoritative_assignments = [
            assignment
            for assignment in _lex_authoritative_answer_assignments(payload)
            if assignment.schema is not None
        ]
        completeness_payload = (
            payload if len(authoritative_assignments) >= 2 else normalized_payload
        )
        if (
            not _answer_candidate_is_complete(completeness_payload)
            and not currency_scalar
        ):
            # A questioned or truncated later answer invalidates stale older
            # candidates, but uncertainty must never be promoted to an answer
            # merely because the model repeats it.  With no later complete
            # candidate the caller therefore scores the raw text fail-closed.
            if clause.rstrip().endswith(("?", "？")) and (
                _answer_cue_payload_is_concrete(
                    text,
                    match.start(),
                    match.end(),
                    end,
                    normalized_payload=normalized_payload,
                )
            ):
                latest_semantic_cue_start = match.start()
                latest_semantic_cue_strength = 100
                semantic_boundary_blocks_recovery = True
            continue
        is_concrete = _answer_cue_payload_is_concrete(
            text,
            match.start(),
            match.end(),
            end,
            normalized_payload=normalized_payload,
        )
        if structured_option_label is not None:
            is_concrete = True
        if not is_concrete and source_stance == "adopted":
            adopted_payload = identity_payload.lstrip("$`*_ ")
            is_concrete = (
                _CONCRETE_MATH_ANSWER_PREFIX_RE.match(adopted_payload) is not None
            )
        if (
            not is_concrete
            and cue_strength >= 4
            and (
                _CONCRETE_MATH_ANSWER_ANYWHERE_RE.search(normalized_payload)
                is not None
            )
            and (
                not _uses_block_layout
                or _explicit_payload_has_structured_math_lead(normalized_payload)
            )
        ):
            is_concrete = True
        is_format_instruction = re.match(
            r"(?i)\s*presented\s+as\b", identity_payload
        ) is not None
        if is_format_instruction:
            latest_semantic_cue_start = match.start()
            latest_semantic_cue_strength = 100
            semantic_boundary_blocks_recovery = True
            continue
        if (
            is_concrete
            or _answer_cue_is_strong_boundary(
                text, match.start(), match.group()
            )
        ) and (
            semantic_boundary_blocks_recovery
            or cue_strength >= latest_semantic_cue_strength
        ):
            latest_semantic_cue_start = match.start()
            latest_semantic_cue_strength = cue_strength
            semantic_boundary_blocks_recovery = False
        if not is_concrete:
            continue
        suffix = text[end:]
        if _answer_candidate_is_retracted(suffix):
            continue
        clause_boxes = _boxed_answer_candidates(clause)
        preserve_display_block = bool(
            _uses_block_layout
            and _answer_block_has_display_wrapper(normalized_payload)
        )
        if len(clause_boxes) > 1 or (
            preserve_display_block and clause_boxes
        ):
            suppressed_component_box_starts.update(
                match.start() + box.start for box in clause_boxes
            )
        if len(clause_boxes) == 1 and not preserve_display_block:
            scoring_text = clause_boxes[-1].scoring_text
            candidate_content = clause_boxes[-1].content
        elif clause_boxes:
            # A multi-part explicit answer may be a tuple, vector, set,
            # interval, matrix, or several independent values.  Preserve the
            # complete committed clause so math_verify sees its structure;
            # never reduce it to the last box or synthesize a collection.
            scoring_text = clause
            candidate_content = (
                clause_boxes[-1].content
                if len(clause_boxes) == 1
                else payload
            )
        elif minimal_scalar is not None:
            scalar_for_scoring = re.sub(
                r"^(?:\\\$|\$)\s*", "", minimal_scalar
            )
            scoring_text = f"{match.group().rstrip()} {scalar_for_scoring}"
            candidate_content = minimal_scalar
        else:
            scoring_text = clause
            candidate_content = payload
        candidates.append(
            _AnswerCandidate(
                match.start(),
                end,
                scoring_text,
                candidate_content,
                cue_strength,
                conflicting=_multi_box_identities_conflict(
                    payload,
                    _boxed_answer_candidates(payload),
                ),
                explicit=True,
            )
        )
    if suppressed_component_box_starts:
        candidates = [
            candidate
            for candidate in candidates
            if not (
                candidate.start in suppressed_component_box_starts
                and candidate.scoring_text.lstrip().startswith("\\boxed{")
            )
        ]
    if latest_semantic_cue_start >= 0:
        candidates = [
            candidate
            for candidate in candidates
            if candidate.start >= latest_semantic_cue_start
            or candidate.strength > latest_semantic_cue_strength
        ]
    terminal_result = _terminal_complete_result_candidate(text)
    if terminal_result is not None and not semantic_boundary_blocks_recovery:
        candidates.append(terminal_result)
    # A boxed/final/answer-labelled candidate remains more authoritative than
    # an adjacent unlabelled result sentence in optional verification.  Within
    # the same evidence strength, the latest complete candidate wins.
    return sorted(candidates, key=lambda item: (item.strength, item.start, item.end))


def _answer_evidence_is_contextual(
    text: str, *, start: int, end: int
) -> bool:
    """Return whether answer evidence is quoted or conditional."""

    source_stance = _source_evidence_stance(text, start=start, end=end)
    if _position_is_quoted(text, min(len(text), start + 1)):
        return source_stance != "adopted"
    boundary = max(
        text.rfind(mark, 0, start) for mark in (".", "!", "?", ";", "\n")
    )
    local_context = text[boundary + 1 : end]
    if source_stance == "adopted":
        return False
    return bool(
        _CONDITIONAL_ANSWER_SUFFIX_RE.search(local_context)
        or source_stance in {"reported", "rejected"}
        or _CONTEXTUAL_EVIDENCE_SOURCE_RE.search(local_context)
    )


def _terminal_bare_answer_candidate(
    text: str, *, minimum_start: int
) -> _AnswerCandidate | None:
    """Return a complete terminal scalar/label after an authority boundary."""

    tail = text[minimum_start:]
    segment_start = 0
    separators = list(re.finditer(r"[.!?](?:\s+|$)|\n+", tail))
    for separator in reversed(separators):
        if tail[separator.end() :].strip():
            segment_start = separator.end()
            break
    raw_segment = tail[segment_start:]
    leading = len(raw_segment) - len(raw_segment.lstrip())
    value = raw_segment.strip().rstrip(".!\u3002\uff01").strip()
    if (
        not value
        or len(value) > 240
        or _BARE_TERMINAL_ANSWER_RE.fullmatch(value) is None
        or not _answer_candidate_is_complete(value)
    ):
        return None
    start = minimum_start + segment_start + leading
    return _AnswerCandidate(start, len(text), value, value, 6, explicit=True)


def _postposed_terminal_bridge_is_authoritative(bridge: str) -> bool:
    """Classify the discourse immediately leading into a terminal answer.

    Whitespace and display delimiters are presentation, so their raw length is
    irrelevant.  If prose intervenes, only its final discourse clause decides:
    an explicit recalculation/revision/conclusion licenses a new terminal
    event, while intermediate/example/substitution language blocks it.
    """

    presentation_normalized = _LATEX_SPACING_COMMAND_RE.sub("", bridge)
    if _PRESENTATION_ONLY_BRIDGE_RE.fullmatch(presentation_normalized) is not None:
        return True
    normalized = re.sub(r"[ \t\r\f\v]+", " ", bridge).strip()
    clauses = [
        clause.strip()
        for clause in re.split(r"(?:[.!?;]\s+|\n+)", normalized)
        if clause.strip()
    ]
    if not clauses:
        return True
    terminal_clause = clauses[-1]
    if _INCIDENTAL_TERMINAL_BOX_CONTEXT_RE.search(terminal_clause):
        return False
    if _BARE_REPLACEMENT_CUE_RE.search(terminal_clause) is not None:
        return True
    if _POSTPOSED_CONCLUSIVE_MARKER_RE.search(terminal_clause) is not None:
        return True
    if _POSTPOSED_CHANGE_EVENT_RE.search(terminal_clause) is not None:
        return True
    return bool(
        _POSTPOSED_REEVALUATION_EVENT_RE.search(terminal_clause)
        and (
            _CONCLUSIVE_RESULT_VERB_RE.search(terminal_clause)
            or _POSTPOSED_RESULT_PREDICATE_RE.search(terminal_clause)
        )
    )


def _postposed_terminal_answer_candidate(
    text: str,
    candidates: Iterable[_AnswerCandidate],
) -> _AnswerCandidate | None:
    """Promote a complete terminal answer that follows an earlier commitment.

    A model sometimes emits ``Final answer: 54`` and then completes with a
    standalone ``\\boxed{55}`` or bare ``55``.  That last terminal event is a
    real chronological replacement even without another answer label.  The
    promotion is deliberately local to the end of the completion: a box with
    later explanatory prose, or one explicitly described as an intermediate
    substitution, remains incidental and cannot displace the commitment.
    """

    materialized = list(candidates)
    committed = [
        candidate
        for candidate in materialized
        if candidate.strength >= _COMMITTED_ANSWER_STRENGTH
    ]
    if not committed:
        return None
    anchor = max(committed, key=lambda item: (item.start, item.strength, item.end))
    terminal: list[_AnswerCandidate] = []

    for boxed in _boxed_answer_candidates(text):
        if boxed.start < anchor.end or not _answer_candidate_is_terminal(text, boxed):
            continue
        bridge = text[anchor.end:boxed.start]
        if _answer_evidence_is_contextual(text, start=boxed.start, end=boxed.end):
            continue
        clause_start = max(
            text.rfind(mark, anchor.end, boxed.start)
            for mark in (".", "!", "?", ";", "\n")
        )
        context_start = max(anchor.end, clause_start + 1)
        clause_boxes = _boxed_answer_candidates(text[context_start:])
        if len(clause_boxes) != 1:
            continue
        if not _postposed_terminal_bridge_is_authoritative(bridge):
            continue
        terminal.append(
            _AnswerCandidate(
                boxed.start,
                boxed.end,
                boxed.scoring_text,
                boxed.content,
                6,
                explicit=True,
            )
        )

    bare = _terminal_bare_answer_candidate(text, minimum_start=anchor.end)
    if (
        bare is not None
        and bare.start > anchor.start
        and _postposed_terminal_bridge_is_authoritative(
            text[anchor.end:bare.start]
        )
    ):
        terminal.append(bare)
    if not terminal:
        return None
    return max(terminal, key=lambda item: (item.start, item.end))


def _scan_answer_evidence(text: str) -> _AnswerEvidenceScan:
    """Resolve answer evidence into candidate/invalidated/none.

    Explicit corrections and retractions create an authority boundary.  Only
    complete, non-contextual candidates at or after the latest boundary may be
    scored.  This prevents a symbolic parser or Judge from resurrecting an old
    answer that the model explicitly withdrew.
    """

    text, serialized_conflict = _unwrap_serialized_answer_text_with_status(text)
    if serialized_conflict:
        return _AnswerEvidenceScan(
            state=_ANSWER_EVIDENCE_INVALIDATED,
            text=text,
        )
    # All semantic passes operate on the same visible projection.  Attributes,
    # comments, and tag names are presentation metadata and must never create
    # assignments, boxes, alternatives, or MCQ labels.  Projecting once here
    # also keeps every candidate/boundary offset internally consistent.
    text = _normalize_answer_html_markup(
        text,
        preserve_field_boundaries=True,
    )
    contextual_evidence = False
    contextual_boundary = -1
    candidates: list[_AnswerCandidate] = []
    for candidate in _explicit_answer_candidates(text):
        if _answer_evidence_is_contextual(
            text,
            start=candidate.start,
            end=candidate.end,
        ):
            contextual_evidence = True
            contextual_boundary = max(contextual_boundary, candidate.end)
            continue
        candidates.append(candidate)

    for cue in _BARE_REPLACEMENT_CUE_RE.finditer(text):
        is_answer_event, payload, end = _replacement_event_payload(
            text,
            cue_start=cue.start(),
            cue_end=cue.end(),
        )
        if not is_answer_event:
            continue
        if _answer_evidence_is_contextual(text, start=cue.start(), end=end):
            contextual_evidence = True
            contextual_boundary = max(contextual_boundary, end)
            continue
        if (
            _answer_candidate_is_complete(payload)
            and _CONCRETE_MATH_ANSWER_PREFIX_RE.match(payload) is not None
        ):
            candidates.append(
                _AnswerCandidate(
                    cue.start(),
                    end,
                    payload,
                    payload,
                    5,
                    explicit=True,
                )
            )
    candidates.sort(key=lambda item: (item.strength, item.start, item.end))

    boundaries = _answer_authority_boundaries(text)
    unresolved_boundary = -1
    for cue in _EXPLICIT_ANSWER_CUE_RE.finditer(text):
        end, uses_block_layout = _explicit_answer_cue_span(
            text,
            cue_start=cue.start(),
            cue_end=cue.end(),
        )
        if _answer_evidence_is_contextual(text, start=cue.start(), end=end):
            contextual_evidence = True
            contextual_boundary = max(contextual_boundary, end)
            continue
        payload = _normalize_answer_block_payload(
            # Do not erase a trailing assignment/component delimiter before
            # the structural completeness gate sees it.
            text[cue.end() : end].strip(" \t\r\n")
        )
        if payload.rstrip().endswith(("?", "\uff1f")):
            boundaries.append(cue.start())
        has_explicit_candidate = any(
            candidate.explicit and candidate.start == cue.start()
            for candidate in candidates
        )
        if (
            _answer_cue_strength(text, cue.start(), cue.group())
            >= _COMMITTED_ANSWER_STRENGTH
            and not has_explicit_candidate
            and (
                uses_block_layout
                or len(_boxed_answer_candidates(payload)) > 1
                or _answer_block_has_incomplete_named_component_tail(payload)
                or _answer_box_tree_has_conflicting_alternatives(payload)
            )
        ):
            # Every committed block-layout answer cue is an authority barrier,
            # whether the malformed/empty block contains no text or contains
            # non-answer presentation. An inline malformed compound with
            # multiple boxes gets the same protection so its components cannot
            # escape. Generic boxes and stale values may never leap across
            # either; single-value inline incomplete corrections retain their
            # established chronological-recovery semantics.
            unresolved_boundary = max(unresolved_boundary, cue.start())

    regular_boundary = max(boundaries) if boundaries else -1
    latest_boundary = max(regular_boundary, unresolved_boundary)
    terminal_minimum = max(latest_boundary, contextual_boundary)
    if terminal_minimum >= 0 and unresolved_boundary < max(
        regular_boundary,
        contextual_boundary,
    ):
        terminal_candidate = _terminal_bare_answer_candidate(
            text,
            minimum_start=terminal_minimum,
        )
        if terminal_candidate is not None:
            candidates.append(terminal_candidate)
            candidates.sort(key=lambda item: (item.strength, item.start, item.end))
    if latest_boundary >= 0:
        candidates = [
            candidate for candidate in candidates if candidate.start >= latest_boundary
        ]
        if unresolved_boundary == latest_boundary:
            candidates = [candidate for candidate in candidates if candidate.explicit]
        if not candidates:
            return _AnswerEvidenceScan(
                state=_ANSWER_EVIDENCE_INVALIDATED,
                text=text,
                contextual_only=contextual_evidence,
            )

    postposed_terminal = _postposed_terminal_answer_candidate(text, candidates)
    if postposed_terminal is not None:
        candidates.append(postposed_terminal)

    if candidates:
        selected = _select_answer_candidate(candidates)
        assert selected is not None
        if selected.conflicting:
            return _AnswerEvidenceScan(
                state=_ANSWER_EVIDENCE_INVALIDATED,
                text=text,
                contextual_only=contextual_evidence,
            )
        return _AnswerEvidenceScan(
            state=_ANSWER_EVIDENCE_CANDIDATE,
            text=text,
            candidate=selected,
            contextual_only=contextual_evidence,
        )
    return _AnswerEvidenceScan(
        state=_ANSWER_EVIDENCE_NONE,
        text=text,
        contextual_only=contextual_evidence,
    )


def _tail_is_syntactically_incomplete(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return True
    last_box_start = stripped.rfind("\\boxed{")
    if last_box_start >= 0 and not any(
        candidate.start == last_box_start
        for candidate in _boxed_answer_candidates(stripped)
    ):
        return True
    last_line = stripped.rsplit("\n", 1)[-1].strip()
    if last_line in {r"\]", r"\)", "</think>"}:
        return False
    return not _answer_candidate_is_complete(last_line)


def _math_verify_input(
    scoring_text: str, *, recover_incomplete_tail: bool = True
) -> str:
    """Keep math_verify focused on the final-answer region.

    Some RWKV outputs contain long repeated reasoning before or after the final
    answer. Passing the full text into math_verify can trigger very slow
    symbolic comparisons even when the answer is plainly present near the end.
    """

    text = scoring_text.strip("\r\n")
    if recover_incomplete_tail:
        evidence = _scan_answer_evidence(text)
        text = evidence.text
        if evidence.candidate is not None:
            selected = evidence.candidate
            if _tail_is_syntactically_incomplete(text) or _answer_candidate_is_terminal(
                text, selected
            ):
                return selected.scoring_text
    if len(text) <= _ANSWER_WINDOW_TAIL_CHARS:
        return text
    boxed_start = text.rfind("\\boxed{")
    if boxed_start >= 0:
        depth = 0
        for offset, char in enumerate(text[boxed_start:]):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[boxed_start : boxed_start + offset + 1]
        return text[boxed_start : boxed_start + _ANSWER_WINDOW_SUFFIX_CHARS]
    lowered = text.lower()
    marker_index = -1
    for marker in _ANSWER_WINDOW_MARKERS:
        index = lowered.rfind(marker)
        marker_index = max(marker_index, index)
    if marker_index >= 0:
        start = max(0, marker_index - _ANSWER_WINDOW_PREFIX_CHARS)
        end = min(len(text), marker_index + _ANSWER_WINDOW_SUFFIX_CHARS)
        return text[start:end]
    return text[-_ANSWER_WINDOW_TAIL_CHARS:]


def _last_boxed_content(text: str) -> str | None:
    candidates = _boxed_answer_candidates(text)
    return candidates[-1].content if candidates else None


def _canonical_simple_integer(value: str) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    text = text.strip().strip("$").strip()
    if text.startswith("\\(") and text.endswith("\\)"):
        text = text[2:-2].strip()
    if text.startswith("\\[") and text.endswith("\\]"):
        text = text[2:-2].strip()
    text = text.strip("{}[]() ").rstrip(".。").replace(",", "")
    if not _SIMPLE_INTEGER_RE.fullmatch(text):
        return None
    try:
        return str(int(float(text))) if "." in text else str(int(text))
    except ValueError:
        return None


def _normalize_escaped_currency_for_math_verify(value: str) -> str:
    """Remove only presentation-only ``\\$`` before a numeric money value.

    ``math_verify`` treats the escaped dollar sign as an unknown LaTeX token,
    although model answers commonly write ``\\$2`` or ``\\$90`` in otherwise
    unambiguous final-answer prose.  Keep ordinary maths dollars untouched and
    strip only an escaped currency marker immediately followed by a number.
    """

    return re.sub(r"\\\$(?=\s*[+-]?(?:\d|\.\d))", "", value)


def _fast_integer_match(
    reference: str,
    scoring_text: str,
    *,
    recover_incomplete_tail: bool = True,
) -> tuple[bool, str] | None:
    if not _env_flag("RWKV_MATH_FAST_INTEGER_MATCH"):
        return None
    ref_int = _canonical_simple_integer(reference)
    if ref_int is None:
        return None

    verify_text = _math_verify_input(
        scoring_text, recover_incomplete_tail=recover_incomplete_tail
    )
    verify_text = _normalize_escaped_currency_for_math_verify(verify_text)
    candidates: list[str] = []
    boxed = _last_boxed_content(verify_text)
    if boxed is not None:
        candidates.append(boxed)
    candidates.extend(match.group(1) for match in _FINAL_INTEGER_RE.finditer(verify_text))
    for line in reversed(verify_text.splitlines()[-3:]):
        candidates.append(line)
    candidates.append(verify_text)

    for candidate in candidates:
        candidate_int = _canonical_simple_integer(candidate)
        if candidate_int is None:
            continue
        return candidate_int == ref_int, candidate_int
    return None


def _math_verify(
    reference: str,
    scoring_text: str,
    *,
    recover_incomplete_tail: bool = True,
) -> _MathVerifyResult:
    evidence = _scan_answer_evidence(scoring_text)
    if evidence.state == _ANSWER_EVIDENCE_INVALIDATED:
        return _MathVerifyResult(
            passed=False,
            answer="",
            fail_reason="authoritative_answer_invalidated",
        )
    if evidence.state == _ANSWER_EVIDENCE_NONE and evidence.contextual_only:
        return _MathVerifyResult(
            passed=False,
            answer="",
            fail_reason="contextual_answer_only",
        )
    authority_boundaries = _answer_authority_boundaries(scoring_text)
    if evidence.candidate is not None and (
        authority_boundaries
        or evidence.candidate.strength >= _COMMITTED_ANSWER_STRENGTH
    ):
        scoring_text = evidence.candidate.scoring_text
    else:
        scoring_text = evidence.text

    api = _load_math_verify()
    display_answer = _short_text(scoring_text)
    if _is_exact_match(scoring_text, reference):
        return _MathVerifyResult(
            passed=True,
            answer=display_answer,
            fail_reason="",
        )
    fast_integer = _fast_integer_match(
        reference,
        scoring_text,
        recover_incomplete_tail=recover_incomplete_tail,
    )
    if fast_integer is not None:
        passed, answer = fast_integer
        return _MathVerifyResult(
            passed=passed,
            answer=answer,
            fail_reason="" if passed else "integer_mismatch",
        )
    if api is None:
        return _MathVerifyResult(
            passed=False,
            answer=display_answer,
            fail_reason="math_verify_missing",
        )
    parse, verify = api
    try:
        with _math_verify_time_limit():
            gold = parse(_reference_expr(reference))
    except _MathVerifyTimeout:
        return _MathVerifyResult(
            passed=False,
            answer=display_answer,
            fail_reason="reference_parse_timeout",
        )
    except Exception as exc:  # noqa: BLE001
        return _MathVerifyResult(
            passed=False,
            answer=display_answer,
            fail_reason=f"reference_parse_error:{type(exc).__name__}",
        )
    verify_text = _math_verify_input(
        scoring_text, recover_incomplete_tail=recover_incomplete_tail
    )
    verify_text = _normalize_escaped_currency_for_math_verify(verify_text)
    try:
        with _math_verify_time_limit():
            pred = parse(verify_text)
    except _MathVerifyTimeout:
        return _MathVerifyResult(
            passed=False,
            answer=display_answer,
            fail_reason="prediction_parse_timeout",
        )
    except Exception as exc:  # noqa: BLE001
        return _MathVerifyResult(
            passed=False,
            answer=display_answer,
            fail_reason=f"prediction_parse_error:{type(exc).__name__}",
        )
    if pred:
        display_answer = _short_text(_parsed_answer_text(pred))
    try:
        with _math_verify_time_limit():
            outcome = (
                _deterministic_math_verify(gold, pred, verify)
                if pred
                else _DeterministicMathVerifyOutcome(passed=False)
            )
    except _MathVerifyTimeout:
        return _MathVerifyResult(
            passed=False,
            answer=display_answer,
            fail_reason="math_verify_timeout",
        )
    except Exception as exc:  # noqa: BLE001
        return _MathVerifyResult(
            passed=False,
            answer=display_answer,
            fail_reason=f"math_verify_error:{type(exc).__name__}",
        )
    return _MathVerifyResult(
        passed=outcome.passed,
        answer=display_answer,
        fail_reason=(
            ""
            if outcome.passed
            else (
                _SYMBOL_BIJECTION_LIMIT_FAIL_REASON
                if outcome.limit_exceeded
                else "math_verify_false"
            )
        ),
    )


def _answer_candidate_option_labels(
    candidate: _AnswerCandidate,
    labels: set[str],
) -> set[str]:
    """Collect legal labels asserted inside one authoritative answer clause."""

    visible_scoring_text = unicodedata.normalize(
        "NFKC",
        _normalize_answer_html_markup(
            candidate.scoring_text,
            preserve_field_boundaries=False,
        ),
    )
    visible_content = unicodedata.normalize(
        "NFKC",
        _normalize_answer_html_markup(
            candidate.content,
            preserve_field_boundaries=False,
        ),
    )
    asserted: set[str] = set()
    for boxed in _boxed_answer_candidates(visible_scoring_text):
        label = _reference_option_label(boxed.content)
        if label in labels:
            asserted.add(label)
    for match in _EXPLICIT_OPTION_LABEL_RE.finditer(visible_scoring_text):
        label = match.group(1).upper()
        if label in labels:
            asserted.add(label)
    for match in re.finditer(
        r"(?x)(?:"
        r"(?:(?i:option|choice))\s+([A-Z])\b|"
        r"[\(\[\{]\s*([A-Z])\s*[\)\]\}]"
        r")",
        visible_content,
    ):
        label = (match.group(1) or match.group(2)).upper()
        if label in labels:
            asserted.add(label)
    return asserted


def _visible_option_labels(value: str, labels: set[str]) -> set[str]:
    """Collect legal labels from visible text, excluding HTML metadata."""

    visible = unicodedata.normalize(
        "NFKC",
        _normalize_answer_html_markup(
            value,
            preserve_field_boundaries=False,
        ),
    )
    asserted: set[str] = set()
    for boxed in _boxed_answer_candidates(visible):
        label = _reference_option_label(boxed.content)
        if label in labels:
            asserted.add(label)
    for match in _EXPLICIT_OPTION_LABEL_RE.finditer(visible):
        label = match.group(1).upper()
        if label in labels:
            asserted.add(label)
    for match in re.finditer(
        r"(?x)(?:"
        r"(?:(?i:option|choice))\s+([A-Z])\b|"
        r"[\(\[\{]\s*([A-Z])\s*[\)\]\}]"
        r")",
        visible,
    ):
        label = (match.group(1) or match.group(2)).upper()
        if label in labels:
            asserted.add(label)
    # Once the question has proven an MCQ schema, two bare labels joined by
    # an explicit visible alternative are answer assertions rather than
    # incidental prose.  Restricting this path to the relation grammar avoids
    # reinterpreting arbitrary one-letter mathematics as a choice.
    for match in re.finditer(
        r"(?i)(?<![A-Z0-9_])([A-Z])\s+"
        r"(?:or|alternatively|otherwise)\s+"
        r"([A-Z])(?![A-Z0-9_])",
        visible,
    ):
        for group in match.groups():
            label = group.upper()
            if label in labels:
                asserted.add(label)
    return asserted


def _multiple_choice_verify(
    question: str,
    reference: str,
    scoring_text: str,
    *,
    recover_incomplete_tail: bool = True,
) -> _MultipleChoiceVerifyResult | None:
    """Resolve structured MCQ labels/text before symbolic or LLM judging.

    A strong explicit label, or an exact final-answer match to one option,
    makes the verdict deterministic.  A recognized wrong option is also
    conclusive and must not be overturned by a stochastic LLM judge.  If the
    output is ambiguous after a structured schema has been proven, fail closed
    so neither symbolic parsing nor a stochastic Judge can resurrect it.
    """

    reference_label = _reference_option_label(reference)
    raw_markers = _question_option_markers(question)
    marker_chain = _ordered_option_marker_chain(
        raw_markers,
        required_label=reference_label,
    )
    chain_labels = {marker[0] for marker in marker_chain}
    relevant_markers = [
        marker for marker in raw_markers if marker[0] in chain_labels
    ]
    if (
        reference_label is not None
        and marker_chain
        and _option_markers_are_ambiguous(relevant_markers)
    ):
        return _MultipleChoiceVerifyResult(
            result=_MathVerifyResult(
                passed=False,
                answer=_short_text(scoring_text),
                fail_reason="ambiguous_multiple_choice_question",
            ),
            conclusive=True,
        )
    options = _parse_question_options(question, required_label=reference_label)
    if not options or reference_label not in options:
        if reference_label is not None and (
            raw_markers or _OPTION_SCHEMA_HINT_RE.search(question) is not None
        ):
            return _MultipleChoiceVerifyResult(
                result=_MathVerifyResult(
                    passed=False,
                    answer=_short_text(scoring_text),
                    fail_reason="invalid_multiple_choice_question",
                ),
                conclusive=True,
            )
        return None

    evidence = _scan_answer_evidence(scoring_text)
    if evidence.state == _ANSWER_EVIDENCE_INVALIDATED or (
        evidence.state == _ANSWER_EVIDENCE_NONE and evidence.contextual_only
    ):
        return _MultipleChoiceVerifyResult(
            result=_MathVerifyResult(
                passed=False,
                answer="",
                fail_reason="authoritative_multiple_choice_answer_invalidated",
            ),
            conclusive=True,
        )
    authoritative_text = (
        evidence.candidate.scoring_text
        if evidence.candidate is not None
        else evidence.text
    )
    labels = set(options)
    asserted_labels: set[str] = set()
    if evidence.candidate is not None:
        asserted_labels.update(
            _answer_candidate_option_labels(
                evidence.candidate,
                labels,
            )
        )
    for final_value in _final_answer_candidates(
        authoritative_text,
        recover_incomplete_tail=recover_incomplete_tail,
    ):
        asserted_labels.update(_visible_option_labels(final_value, labels))
    if len(asserted_labels) > 1:
        return _MultipleChoiceVerifyResult(
            result=_MathVerifyResult(
                passed=False,
                answer="",
                fail_reason="authoritative_multiple_choice_answer_invalidated",
            ),
            conclusive=True,
        )
    normalized_options: dict[str, str] = {
        label: _comparable_option_text(value) for label, value in options.items()
    }
    if evidence.candidate is not None:
        payload_label = _reference_option_label(evidence.candidate.content)
        if payload_label in labels:
            passed = payload_label == reference_label
            return _MultipleChoiceVerifyResult(
                result=_MathVerifyResult(
                    passed=passed,
                    answer=payload_label,
                    fail_reason="" if passed else "multiple_choice_label_mismatch",
                ),
                conclusive=True,
            )
        normalized_payload = _comparable_option_text(evidence.candidate.content)
        payload_matches = {
            label
            for label, option in normalized_options.items()
            if normalized_payload and normalized_payload == option
        }
        if len(payload_matches) == 1:
            predicted_label = next(iter(payload_matches))
            passed = predicted_label == reference_label
            return _MultipleChoiceVerifyResult(
                result=_MathVerifyResult(
                    passed=passed,
                    answer=predicted_label,
                    fail_reason="" if passed else "multiple_choice_option_mismatch",
                ),
                conclusive=True,
            )
    predicted_label = _explicit_option_label(
        authoritative_text,
        labels,
        recover_incomplete_tail=recover_incomplete_tail,
    )
    if predicted_label is not None:
        passed = predicted_label == reference_label
        return _MultipleChoiceVerifyResult(
            result=_MathVerifyResult(
                passed=passed,
                answer=predicted_label,
                fail_reason="" if passed else "multiple_choice_label_mismatch",
            ),
            conclusive=True,
        )

    matched_labels: set[str] = set()
    for candidate in _final_answer_candidates(
        authoritative_text, recover_incomplete_tail=recover_incomplete_tail
    ):
        normalized = _comparable_option_text(candidate)
        if not normalized:
            continue
        matched_labels.update(
            label for label, option in normalized_options.items() if normalized == option
        )
    if len(matched_labels) == 1:
        predicted_label = next(iter(matched_labels))
        passed = predicted_label == reference_label
        return _MultipleChoiceVerifyResult(
            result=_MathVerifyResult(
                passed=passed,
                answer=predicted_label,
                fail_reason="" if passed else "multiple_choice_option_mismatch",
            ),
            conclusive=True,
        )

    fallback = _math_verify(
        reference,
        authoritative_text,
        recover_incomplete_tail=recover_incomplete_tail,
    )
    # math_verify can still extract a clean scalar/expression while comparing
    # it against the reference *label* (and therefore report false).  Reuse
    # that extracted display answer only when it exactly equals one uniquely
    # parsed option.  This keeps the mapping global and conservative while
    # covering answers such as ``120 adult tickets`` for option ``C) 120``.
    normalized_fallback = _comparable_option_text(fallback.answer)
    fallback_matches = {
        label
        for label, option in normalized_options.items()
        if normalized_fallback and normalized_fallback == option
    }
    if len(fallback_matches) == 1:
        predicted_label = next(iter(fallback_matches))
        passed = predicted_label == reference_label
        return _MultipleChoiceVerifyResult(
            result=_MathVerifyResult(
                passed=passed,
                answer=predicted_label,
                fail_reason="" if passed else "multiple_choice_option_mismatch",
            ),
            conclusive=True,
        )

    return _MultipleChoiceVerifyResult(result=fallback, conclusive=False)


def _judgement_verify(reference: str, scoring_text: str) -> _MathVerifyResult:
    expected = _extract_judgement_label(reference)
    actual = _extract_judgement_label(scoring_text)
    display_answer = actual or _short_text(scoring_text)
    passed = bool(expected and actual and expected == actual)
    if passed:
        fail_reason = ""
    elif actual is None:
        fail_reason = "judgement_label_missing"
    else:
        fail_reason = "judgement_label_mismatch"
    return _MathVerifyResult(
        passed=passed,
        answer=display_answer,
        fail_reason=fail_reason,
    )


def _parsed_answer_text(parsed: Any) -> str:
    if isinstance(parsed, (list, tuple)) and parsed:
        item = parsed[-1]
    else:
        item = parsed
    if isinstance(item, (list, tuple)) and item:
        item = item[-1]
    return str(item)


def _generation_stop_suffixes_for_prompt(prompt: str) -> tuple[str, ...]:
    if "User✿" in prompt or "Bot✿" in prompt:
        return G1H_GENERATION_STOP_SUFFIXES
    return LEGACY_GENERATION_STOP_SUFFIXES


def _clip_generation_sentinels(text: str, *, prompt: str = "") -> str:
    cut = len(text)
    for sentinel in _generation_stop_suffixes_for_prompt(prompt):
        index = text.find(sentinel)
        if index >= 0:
            cut = min(cut, index)
    return text[:cut].rstrip()


def _stage_text(payload: dict[str, Any], stage: int) -> str:
    text = str(payload.get(f"completion{stage}") or "")
    prompt = _stage_prompt(payload, stage)
    if is_naive_nocot_prompt(prompt):
        text = strip_generated_empty_think_closer(text)
    return _clip_generation_sentinels(text, prompt=prompt)


def _stage_prompt(payload: dict[str, Any], stage: int) -> str:
    return str(payload.get(f"prompt{stage}") or "")


def _stage_stop_reason(payload: dict[str, Any], stage: int) -> str:
    return str(payload.get(f"stop_reason{stage}") or "")


def _has_stage(payload: dict[str, Any], stage: int) -> bool:
    return f"completion{stage}" in payload or f"prompt{stage}" in payload


def _has_blank_recovery_stage(payload: dict[str, Any]) -> bool:
    """Return whether a persisted recovery stage exists but generated no answer.

    Stage presence is deliberately structural: legacy Strategy-A-only payloads
    have no stage 2 and retain their historical scoring behaviour.  Once a
    stage-2 prompt/completion is present, however, an empty generation (also
    after clipping model stop sentinels) is an explicit missing prediction,
    not permission to score the reasoning prompt or synthesize an empty box.
    """

    return _has_stage(payload, 2) and not _stage_text(payload, 2).strip()


def _completion_text(payload: dict[str, Any]) -> str:
    return _stage_text(payload, 1)


def _completion_prompt(payload: dict[str, Any]) -> str:
    return _stage_prompt(payload, 1)


def _completion_stop_reason(payload: dict[str, Any]) -> str:
    return _stage_stop_reason(payload, 1)


def _has_strategy_a(payload: dict[str, Any]) -> bool:
    """Return whether a dedicated Strategy-A generation was persisted.

    Presence is structural for the same reason as the recovery-stage check:
    an explicitly persisted Strategy A owns its result even when the model
    generated an empty string.  Legacy one-stage payloads have none of these
    keys and continue to use stage 1 as their Strategy-A result.
    """

    return any(
        key in payload
        for key in (
            "strategy_a_prompt",
            "strategy_a_completion",
            "strategy_a_stop_reason",
        )
    )


def _strategy_a_text(payload: dict[str, Any]) -> str:
    if _has_strategy_a(payload):
        text = str(payload.get("strategy_a_completion") or "")
        prompt = _strategy_a_prompt(payload)
        if is_naive_nocot_prompt(prompt):
            text = strip_generated_empty_think_closer(text)
        return _clip_generation_sentinels(text, prompt=prompt)
    return _completion_text(payload)


def _has_blank_strategy_a(payload: dict[str, Any]) -> bool:
    """Return whether an explicit Strategy A generated no usable answer."""

    return _has_strategy_a(payload) and not _strategy_a_text(payload).strip()


def _strategy_a_prompt(payload: dict[str, Any]) -> str:
    prompt = str(payload.get("strategy_a_prompt") or "")
    return prompt if _has_strategy_a(payload) else _completion_prompt(payload)


def _strategy_a_stop_reason(payload: dict[str, Any]) -> str:
    reason = str(payload.get("strategy_a_stop_reason") or "")
    return reason if _has_strategy_a(payload) else _completion_stop_reason(payload)


def _stage_is_truncated(payload: dict[str, Any], stage: int) -> bool:
    if _stage_stop_reason(payload, stage) in {"max_tokens", "max_length"}:
        return True
    stats = payload.get("stats")
    if isinstance(stats, dict):
        stage_stats = stats.get(f"stage{stage}")
        if isinstance(stage_stats, dict):
            return bool(stage_stats.get("truncated"))
    return False


def _is_truncated(payload: dict[str, Any]) -> bool:
    if _has_stage(payload, 2):
        if _stage_is_truncated(payload, 2):
            return True
        stats = payload.get("stats")
        return isinstance(stats, dict) and bool(stats.get("truncated"))
    if _stage_is_truncated(payload, 1):
        return True
    stats = payload.get("stats")
    return isinstance(stats, dict) and bool(stats.get("truncated"))


def _strategy_a_is_truncated(payload: dict[str, Any]) -> bool:
    if _strategy_a_stop_reason(payload) in {"max_tokens", "max_length"}:
        return True
    stats = payload.get("stats")
    if isinstance(stats, dict):
        strategy_stats = stats.get("strategy_a")
        if isinstance(strategy_stats, dict):
            return bool(strategy_stats.get("truncated"))
    if _has_strategy_a(payload):
        return False
    return _is_truncated(payload)


def _strategy_is_truncated(group: str, payload: dict[str, Any]) -> bool:
    if group == STRATEGY_A:
        return _strategy_a_is_truncated(payload)
    return _is_truncated(payload)


def _think_state(prompt: str, text: str) -> tuple[bool, bool]:
    context = f"{prompt}{text}".lower()
    has_think = "<think" in context
    has_close = "</think>" in context
    return has_think, has_close


def _two_stage_scoring_text(payload: dict[str, Any]) -> str:
    """Return only the authoritative final-stage prompt suffix and output.

    Stage-1 reasoning remains generation context, but it is not a fallback
    answer source once a structural stage 2 exists.  Including it in the text
    passed to math_verify let a truncated stage-2 answer inherit an earlier
    boxed value across the stage boundary.
    """

    prompt1 = _stage_prompt(payload, 1)
    text1 = _stage_text(payload, 1)
    prompt2 = _stage_prompt(payload, 2)
    text2 = _stage_text(payload, 2)
    if not prompt2:
        return text2
    prior_context = f"{prompt1}{text1}"
    # Formal payloads store stage-2 prompts as the delta after stage 1. Accept
    # legacy payloads that stored the already-expanded full prompt by removing
    # that exact prefix before scoring.
    prompt_suffix = (
        prompt2[len(prior_context) :]
        if prompt2.startswith(prior_context)
        else prompt2
    )
    return f"{prompt_suffix}{text2}"


def _has_unclosed_boxed(text: str) -> bool:
    last_box = text.rfind("\\boxed{")
    if last_box < 0:
        return False
    tail = text[last_box + len("\\boxed{") :]
    depth = 1
    for char in tail:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return False
    return depth > 0


def _append_final_cue(text: str) -> str:
    repaired = text.rstrip()
    if REPAIR_FINAL_CUE in repaired[-320:]:
        return repaired
    return f"{repaired}\n{REPAIR_FINAL_CUE}"


def _two_stage_think_state(payload: dict[str, Any]) -> tuple[bool, bool]:
    """Inspect the generated CoT stage, not the synthesized final prompt.

    The final-stage template intentionally contains ``</think>``. Looking at
    that prompt would therefore hide an actually unclosed first-stage think
    block and make the B/C repair decision incorrectly.
    """

    return _think_state(_stage_prompt(payload, 1), _stage_text(payload, 1))


def _repair_two_stage_scoring_text(
    payload: dict[str, Any], text: str, *, group: str
) -> str:
    has_think, has_close = _two_stage_think_state(payload)
    repaired = text.rstrip()
    if has_think and not has_close:
        prior_context = f"{_stage_prompt(payload, 1)}{_stage_text(payload, 1)}"
        final_prompt = _stage_prompt(payload, 2)
        final_text = _stage_text(payload, 2)
        # The persisted stage-2 prompt is normally a delta such as
        # ``Therefore ... \boxed{``.  Close the unfinished reasoning *at the
        # stage boundary*, before that prompt.  Appending ``</think>`` after
        # the generated answer would put synthetic text inside the unclosed
        # box and make math_verify parse e.g. ``2^{99} </think> ...`` as the
        # answer.  Legacy payloads may store the expanded prompt; reduce it to
        # the same suffix before rebuilding the scoring text.
        final_suffix = (
            final_prompt[len(prior_context) :]
            if final_prompt.startswith(prior_context)
            else final_prompt
        )
        if "</think>" in final_suffix.lower():
            repaired = f"{final_suffix}{final_text}".rstrip()
        else:
            repaired = f"</think>{final_suffix}{final_text}".rstrip()
        # Strategy B repairs only the reasoning boundary. Strategy C also
        # closes the syntactic box deliberately prefilled by the final prompt.
        if group == STRATEGY_C and _has_unclosed_boxed(repaired):
            repaired = f"{repaired}}}"
        return repaired

    # C's second repair handles the distinct case where thinking is complete
    # but the answer region itself hit the generation limit. B deliberately
    # leaves this case untouched.
    if group == STRATEGY_C and _stage_is_truncated(payload, 2):
        repaired = _append_final_cue(repaired)
        if _has_unclosed_boxed(repaired):
            repaired = f"{repaired}}}"
        return repaired
    if group == STRATEGY_C and _has_unclosed_boxed(repaired):
        # The final-stage TOML intentionally pre-fills ``\\boxed{``. Closing
        # that syntactic prefix is a format repair, not a new answer source.
        return f"{repaired}}}"
    return text


def _strategy_scoring_text(group: str, payload: dict[str, Any]) -> str:
    if group == STRATEGY_A:
        return _strategy_a_text(payload)
    if _has_stage(payload, 2):
        two_stage_text = _two_stage_scoring_text(payload)
        if group == STRATEGY_B:
            return _repair_two_stage_scoring_text(
                payload, two_stage_text, group=STRATEGY_B
            )
        if group == STRATEGY_C:
            return _repair_two_stage_scoring_text(
                payload, two_stage_text, group=STRATEGY_C
            )

    text = _strategy_a_text(payload)
    prompt = _strategy_a_prompt(payload)
    has_think, has_close = _think_state(prompt, text)
    unclosed_think = has_think and not has_close
    truncated = _strategy_a_is_truncated(payload)

    if group == STRATEGY_B and unclosed_think:
        return f"{text.rstrip()}\n</think>"
    if group == STRATEGY_C and unclosed_think:
        return f"{text.rstrip()}\n</think>\n{REPAIR_FINAL_CUE}"
    if group == STRATEGY_C and truncated and (not has_think or has_close):
        return f"{text.rstrip()}\n{REPAIR_FINAL_CUE}"
    return text


def _strategy_judgement_text(group: str, payload: dict[str, Any]) -> str:
    """Return generated judgement text without including source prompts.

    Math parsing sometimes needs the synthesized prompt suffix (for example a
    pre-opened ``\\boxed{``), but judgement extraction must never scan the
    user prompt: answer-judge prompts explicitly list both ``Judgement: Yes``
    and ``Judgement: No``.  For two-stage B/C evaluation, the final recovery
    output is authoritative, including when it is empty.  The scorer maps that
    empty output to ``missing_recovery_prediction`` instead of silently using
    stage-1 reasoning.  Strategy A always uses its own generated completion.
    """

    if group == STRATEGY_A:
        return _strategy_a_text(payload)
    if _has_stage(payload, 2):
        return _stage_text(payload, 2)
    return _strategy_a_text(payload)


def _stop_rate(payloads: list[dict[str, Any]], *, group: str) -> float:
    if not payloads:
        return 0.0
    return sum(1 for payload in payloads if _strategy_is_truncated(group, payload)) / len(payloads)


def score_free_response_strategy(
    group: str,
    payload: dict[str, Any],
    *,
    sample_index: int,
    repeat_index: int,
    question: str,
    reference: str,
) -> _ScoredCompletion:
    if _has_blank_strategy_a(payload) and (
        group == STRATEGY_A or not _has_stage(payload, 2)
    ):
        return _ScoredCompletion(
            source_payload=payload,
            sample_index=sample_index,
            repeat_index=repeat_index,
            question=question,
            reference=reference,
            scoring_text="",
            display_answer="",
            math_passed=False,
            final_passed=False,
            fail_reason=MISSING_STRATEGY_A_PREDICTION,
            judge_eligible=False,
        )
    if group in {STRATEGY_B, STRATEGY_C} and _has_blank_recovery_stage(payload):
        return _ScoredCompletion(
            source_payload=payload,
            sample_index=sample_index,
            repeat_index=repeat_index,
            question=question,
            reference=reference,
            scoring_text="",
            display_answer="",
            math_passed=False,
            final_passed=False,
            fail_reason=MISSING_RECOVERY_PREDICTION,
            judge_eligible=False,
        )
    if _is_judgement_reference(reference):
        scoring_text = _strategy_judgement_text(group, payload)
        verify_result = _judgement_verify(reference, scoring_text)
        judge_eligible = True
    else:
        scoring_text = _strategy_scoring_text(group, payload)
        # A raw completion may be cut off after already stating a complete
        # answer.  Two-stage B/C, however, have an authoritative recovery
        # stage and must never fall back across that stage boundary to an
        # earlier reasoning answer.
        recover_incomplete_tail = group == STRATEGY_A or not _has_stage(payload, 2)
        mcq_result = _multiple_choice_verify(
            question,
            reference,
            scoring_text,
            recover_incomplete_tail=recover_incomplete_tail,
        )
        if mcq_result is None:
            verify_result = _math_verify(
                reference,
                scoring_text,
                recover_incomplete_tail=recover_incomplete_tail,
            )
            judge_eligible = (
                verify_result.fail_reason
                not in _AUTHORITATIVE_ANSWER_FAIL_REASONS
            )
        else:
            verify_result = mcq_result.result
            judge_eligible = not mcq_result.conclusive
    return _ScoredCompletion(
        source_payload=payload,
        sample_index=sample_index,
        repeat_index=repeat_index,
        question=question,
        reference=reference,
        scoring_text=scoring_text,
        display_answer=verify_result.answer,
        math_passed=verify_result.passed,
        final_passed=verify_result.passed,
        fail_reason=verify_result.fail_reason,
        judge_eligible=judge_eligible,
    )


def evaluate_free_response(
    completions: Iterable[dict] | str | Path,
    *,
    dataset_path: str | Path,
    judge: LLMJudge | None = None,
    primary_only: bool = False,
    primary_group: str | None = None,
    math_verify_retry_timeout_s: float | None = 15.0,
) -> FreeResponseEvaluation:
    """Evaluate full-generation free-response completions.

    When ``math_verify_retry_timeout_s`` is set, only rows whose first
    deterministic verification attempt timed out are retried, one row and one
    strategy at a time, with that deadline.  Retry provenance is returned in
    ``math_verify_retry_stats_by_group``.  A retry that still times out raises
    :class:`UnresolvedMathVerifyTimeoutError` before any eval payload can be
    returned to a persistence caller.  A bounded symbol-bijection search that
    cannot be exhausted likewise raises
    :class:`UnresolvedMathVerifySymbolBijectionError`; it is never persisted as
    a wrong answer.  Passing ``None`` explicitly disables timeout retry and is
    reserved for controlled diagnostics and historical replay.
    """

    if math_verify_retry_timeout_s is not None and math_verify_retry_timeout_s <= 0:
        raise ValueError("math_verify_retry_timeout_s must be positive")

    dataset = list(JsonlFreeAnswerLoader(str(dataset_path)))
    completion_payloads = list(_iter_completions(completions))
    references = [resolve_reference_answer(record) for record in dataset]
    judgement_reference_count = sum(1 for reference in references if _is_judgement_reference(reference))
    judgement_label_dataset = bool(dataset) and judgement_reference_count == len(dataset)
    if not judgement_label_dataset and _load_math_verify() is None:
        raise RuntimeError("free-response evaluation requires math-verify; run `uv sync` after updating uv.lock.")

    if primary_group is None:
        primary_group = STRATEGY_C if judgement_label_dataset else STRATEGY_A
    if primary_group not in STRATEGY_GROUPS:
        raise ValueError(f"unsupported free-response primary group: {primary_group!r}")
    groups_to_score = (primary_group,) if primary_only else STRATEGY_GROUPS
    grouped: dict[str, list[_ScoredCompletion]] = {group: [] for group in groups_to_score}
    math_verify_retry_stats_by_group: dict[str, dict[str, object]] = {}
    if math_verify_retry_timeout_s is not None:
        math_verify_retry_stats_by_group = {
            group: {
                "attempted_count": 0,
                "resolved_count": 0,
                "unresolved_count": 0,
                "rows": [],
            }
            for group in groups_to_score
        }

    def apply_judge(group: str) -> None:
        if judge is None:
            return
        records = grouped[group]
        judge_inputs: list[tuple[str, str, str]] = []
        judge_indices: list[int] = []
        for idx, record in enumerate(records):
            if record.final_passed or not record.judge_eligible:
                continue
            judge_inputs.append((record.question, record.reference, record.display_answer))
            judge_indices.append(idx)
        if not judge_inputs:
            # A root Judge task still needs durable evidence of the exact
            # Judge protocol when deterministic matching resolves every row.
            # Persist a zero-call, fingerprinted run instead of making the
            # audit infer protocol correctness from the absence of requests.
            judge_stats_by_group[group] = LLMJudgeStats(
                total=0,
                protocol=llm_judge_protocol(judge.config),
            ).as_dict()
            return
        judged_flags = judge.judge(judge_inputs)
        stats = judge.last_run_stats
        if stats is not None:
            judge_stats_by_group[group] = stats.as_dict()
        for idx, judged in zip(judge_indices, judged_flags, strict=True):
            record = records[idx]
            record.final_passed = bool(judged)
            if judged:
                record.fail_reason = ""
            else:
                record.fail_reason = (
                    f"{record.fail_reason};judge_false" if record.fail_reason else "judge_false"
                )

    def score_group(
        group: str,
        payload: dict[str, Any],
        *,
        sample_index: int,
        repeat_index: int,
        question: str,
        reference: str,
    ) -> _ScoredCompletion:
        scored = score_free_response_strategy(
            group,
            payload,
            sample_index=sample_index,
            repeat_index=repeat_index,
            question=question,
            reference=reference,
        )
        if scored.fail_reason == _SYMBOL_BIJECTION_LIMIT_FAIL_REASON:
            raise UnresolvedMathVerifySymbolBijectionError(
                group=group,
                sample_index=sample_index,
                repeat_index=repeat_index,
            )
        if (
            math_verify_retry_timeout_s is None
            or not _is_math_verify_timeout_reason(scored.fail_reason)
        ):
            return scored

        first_fail_reason = scored.fail_reason
        with _temporary_math_verify_timeout(math_verify_retry_timeout_s):
            retried = score_free_response_strategy(
                group,
                payload,
                sample_index=sample_index,
                repeat_index=repeat_index,
                question=question,
            reference=reference,
        )
        if retried.fail_reason == _SYMBOL_BIJECTION_LIMIT_FAIL_REASON:
            raise UnresolvedMathVerifySymbolBijectionError(
                group=group,
                sample_index=sample_index,
                repeat_index=repeat_index,
            )
        unresolved = _is_math_verify_timeout_reason(retried.fail_reason)
        stats = math_verify_retry_stats_by_group[group]
        stats["attempted_count"] = int(stats["attempted_count"]) + 1
        outcome_key = "unresolved_count" if unresolved else "resolved_count"
        stats[outcome_key] = int(stats[outcome_key]) + 1
        rows = stats["rows"]
        assert isinstance(rows, list)
        rows.append(
            {
                "sample_index": sample_index,
                "repeat_index": repeat_index,
                "first_fail_reason": first_fail_reason,
                "retry_fail_reason": retried.fail_reason,
                "resolved": not unresolved,
            }
        )
        if unresolved:
            raise UnresolvedMathVerifyTimeoutError(
                group=group,
                sample_index=sample_index,
                repeat_index=repeat_index,
                first_fail_reason=str(first_fail_reason),
                retry_fail_reason=str(retried.fail_reason),
            )
        return retried

    def inherit_from_a(a_record: _ScoredCompletion) -> _ScoredCompletion:
        return _ScoredCompletion(
            source_payload=a_record.source_payload,
            sample_index=a_record.sample_index,
            repeat_index=a_record.repeat_index,
            question=a_record.question,
            reference=a_record.reference,
            scoring_text=a_record.scoring_text,
            display_answer=a_record.display_answer,
            math_passed=a_record.math_passed,
            final_passed=a_record.final_passed,
            fail_reason=a_record.fail_reason,
            judge_eligible=a_record.judge_eligible,
        )

    judge_stats_by_group: dict[str, dict[str, object]] = {}
    record_contexts: list[tuple[dict[str, Any], int, int, str, str]] = []

    for payload in completion_payloads:
        sample_index = strict_nonneg_int(payload.get("sample_index"), "sample_index")
        repeat_index = strict_nonneg_int(payload.get("repeat_index"), "repeat_index")
        if sample_index < 0 or sample_index >= len(dataset):
            question = ""
            reference = ""
        else:
            record = dataset[sample_index]
            question = record.question
            reference = resolve_reference_answer(record)

        record_contexts.append((payload, sample_index, repeat_index, question, reference))
        group = primary_group if primary_only else STRATEGY_A
        grouped[group].append(
            score_group(
                group,
                payload,
                sample_index=sample_index,
                repeat_index=repeat_index,
                question=question,
                reference=reference,
            )
        )

    apply_judge(primary_group if primary_only else STRATEGY_A)

    if not primary_only:
        for group in (STRATEGY_B, STRATEGY_C):
            for idx, (payload, sample_index, repeat_index, question, reference) in enumerate(record_contexts):
                a_record = grouped[STRATEGY_A][idx]
                if a_record.final_passed:
                    grouped[group].append(inherit_from_a(a_record))
                    continue
                grouped[group].append(
                    score_group(
                        group,
                        payload,
                        sample_index=sample_index,
                        repeat_index=repeat_index,
                        question=question,
                        reference=reference,
                    )
                )
            apply_judge(group)

    rows_by_group: dict[str, list[tuple[int, int, bool]]] = {}
    metrics_by_group: dict[str, dict[str, float]] = {}
    eval_payloads: list[dict] = []
    eval_payloads_by_group: dict[str, list[dict]] = {}
    samples = len(completion_payloads)
    for group in groups_to_score:
        records = grouped[group]
        group_payloads: list[dict] = []
        rows = [
            (record.sample_index, record.repeat_index, bool(record.final_passed))
            for record in records
        ]
        rows_by_group[group] = rows
        exact_accuracy = (
            sum(1 for record in records if record.math_passed) / samples if samples else 0.0
        )
        stop_rate = _stop_rate(completion_payloads, group=group)
        metrics: dict[str, float] = {
            "exact_accuracy": exact_accuracy,
            "stop_rate": stop_rate,
        }
        if judge is not None:
            metrics["judge_accuracy"] = (
                sum(1 for record in records if record.final_passed) / samples if samples else 0.0
            )
        metrics_by_group[group] = metrics
        for record in records:
            group_payloads.append(
                make_eval_payload(
                    record.source_payload,
                    is_passed=record.final_passed,
                    fail_reason=record.fail_reason,
                    answer=record.display_answer,
                    ref_answer=record.reference,
                )
            )
        eval_payloads_by_group[group] = group_payloads
        if group == primary_group:
            eval_payloads.extend(group_payloads)

    return FreeResponseEvaluation(
        metrics_by_group=metrics_by_group,
        rows_by_group=rows_by_group,
        samples=samples,
        payloads=eval_payloads,
        payloads_by_group=eval_payloads_by_group,
        judge_stats_by_group=judge_stats_by_group,
        math_verify_retry_stats_by_group=math_verify_retry_stats_by_group,
        primary_group=primary_group,
    )


def build_grouped_metrics_payload(
    evaluation: FreeResponseEvaluation,
    *,
    pass_k: tuple[int, ...],
    avg_k: tuple[NumericK, ...],
    report_pass_k: tuple[int, ...] = (),
    report_avg_k: tuple[NumericK, ...] = (),
) -> tuple[dict[str, object], dict[str, object]]:
    group = evaluation.primary_group
    rows = evaluation.rows_by_group.get(group, [])
    metrics_payload: dict[str, object] = dict(evaluation.metrics_by_group.get(group, {}))
    strategy_metrics: dict[str, dict[str, float]] = {}
    pass_metrics_all = compute_pass_at_k(rows, pass_k)
    avg_metrics_all = compute_avg_at_k(rows, avg_k)

    pass_payload = filter_metrics_by_k(pass_metrics_all, report_pass_k, "pass@")
    if report_pass_k and not pass_payload:
        pass_payload = pass_metrics_all or {}
    if pass_payload:
        metrics_payload.update(pass_payload)

    avg_payload = filter_metrics_by_k(avg_metrics_all, report_avg_k, "avg@")
    if report_avg_k and not avg_payload:
        avg_payload = avg_metrics_all or {}
    if avg_payload:
        metrics_payload.update(avg_payload)

    for strategy in STRATEGY_GROUPS:
        strategy_rows = evaluation.rows_by_group.get(strategy, [])
        group_metrics = dict(evaluation.metrics_by_group.get(strategy, {}))
        group_pass_all = compute_pass_at_k(strategy_rows, pass_k)
        group_avg_all = compute_avg_at_k(strategy_rows, avg_k)

        group_pass_payload = filter_metrics_by_k(group_pass_all, report_pass_k, "pass@")
        if report_pass_k and not group_pass_payload:
            group_pass_payload = group_pass_all or {}
        if group_pass_payload:
            group_metrics.update(group_pass_payload)

        group_avg_payload = filter_metrics_by_k(group_avg_all, report_avg_k, "avg@")
        if report_avg_k and not group_avg_payload:
            group_avg_payload = group_avg_all or {}
        if group_avg_payload:
            group_metrics.update(group_avg_payload)

        strategy_metrics[strategy] = group_metrics
    metrics_payload["strategy_metrics"] = strategy_metrics
    metrics_payload["strategy_diagnostics"] = _build_strategy_diagnostics(evaluation)

    task_details: dict[str, object] = {}
    primary_judge_stats = evaluation.judge_stats_by_group.get(group)
    if primary_judge_stats:
        # Scores are the durable database artifact.  ``task_details`` remains
        # useful to JSONL consumers, but SqlEvalDbRepository intentionally
        # persists only ``metrics``; keep the judge transport/parse evidence
        # here as well so an accepted score can be audited from the database.
        metrics_payload["judge_stats"] = dict(primary_judge_stats)
        task_details["judge_stats"] = primary_judge_stats
    if pass_metrics_all and pass_payload != pass_metrics_all:
        task_details["pass_curve"] = pass_metrics_all
    if avg_metrics_all and avg_payload != avg_metrics_all:
        task_details["avg_curve"] = avg_metrics_all
    return metrics_payload, task_details


def attach_strategy_task_ids(metrics_payload: dict[str, object], task_ids: dict[str, int | str]) -> dict[str, object]:
    metrics_payload["strategy_task_ids"] = {key: int(value) for key, value in task_ids.items()}
    return metrics_payload


def _build_strategy_diagnostics(evaluation: FreeResponseEvaluation) -> dict[str, dict[str, float]]:
    primary_records = _records_by_key(evaluation.payloads_by_group.get(evaluation.primary_group, []))
    diagnostics: dict[str, dict[str, float]] = {}
    for strategy in STRATEGY_GROUPS:
        if strategy == evaluation.primary_group:
            continue
        rows = evaluation.payloads_by_group.get(strategy, [])
        changed = 0
        rescued = 0
        harmed = 0
        compared = 0
        for payload in rows:
            key = (
                strict_nonneg_int(payload.get("sample_index"), "sample_index"),
                strict_nonneg_int(payload.get("repeat_index"), "repeat_index"),
            )
            primary = primary_records.get(key)
            if primary is None:
                continue
            compared += 1
            primary_answer = _normalize_text(str(primary.get("answer") or ""))
            strategy_answer = _normalize_text(str(payload.get("answer") or ""))
            if primary_answer != strategy_answer:
                changed += 1
            primary_passed = bool(primary.get("is_passed"))
            strategy_passed = bool(payload.get("is_passed"))
            if not primary_passed and strategy_passed:
                rescued += 1
            if primary_passed and not strategy_passed:
                harmed += 1
        denominator = compared or 1
        diagnostics[strategy] = {
            "changed_answer_rate": changed / denominator,
            "rescued_rate": rescued / denominator,
            "harmed_rate": harmed / denominator,
        }
    return diagnostics


def _records_by_key(payloads: list[dict]) -> dict[tuple[int, int], dict]:
    records: dict[tuple[int, int], dict] = {}
    for payload in payloads:
        key = (
            strict_nonneg_int(payload.get("sample_index"), "sample_index"),
            strict_nonneg_int(payload.get("repeat_index"), "repeat_index"),
        )
        records[key] = payload
    return records


__all__ = [
    "DEFAULT_LLM_JUDGE_PROMPT_TEMPLATE",
    "G1H_GENERATION_STOP_SUFFIXES",
    "LLM_JUDGE_PROTOCOL_VERSION",
    "LLM_JUDGE_RESPONSE_CONTRACT",
    "LLMJudge",
    "LLMJudgeConfig",
    "LLMJudgeStats",
    "LEGACY_GENERATION_STOP_SUFFIXES",
    "MISSING_RECOVERY_PREDICTION",
    "REPAIR_FINAL_CUE",
    "STRATEGY_A",
    "STRATEGY_B",
    "STRATEGY_C",
    "STRATEGY_GROUPS",
    "STRATEGY_LABELS",
    "FreeResponseEvaluation",
    "attach_strategy_task_ids",
    "build_grouped_metrics_payload",
    "compute_avg_at_k",
    "compute_pass_at_k",
    "evaluate_free_response",
    "llm_judge_prompt_sha256",
    "llm_judge_protocol",
    "llm_judge_protocol_fingerprint",
    "llm_judge_protocol_stats_reasons",
    "resolve_reference_answer",
    "score_free_response_strategy",
]
