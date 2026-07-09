"""Shared utilities: API clients, data I/O, constants.

Reads credentials from environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY),
auto-loaded from a .env file in the repo root if present. If a key is still
missing, falls back to MOCK MODE so the full pipeline can be dry-run and
verified end to end before real API keys are supplied.
"""

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    # override=True: .env is the source of truth for this project. Without it,
    # a stale/corrupted value already exported in the shell (e.g. from an
    # earlier `export $(cat .env | xargs)`) would silently take precedence.
    load_dotenv(REPO_ROOT / ".env", override=True)
except ImportError:
    pass  # python-dotenv not installed; fall back to already-exported env vars

DATA_DIR = REPO_ROOT / "data"

# Defensive strip: guards against stray whitespace/newlines in a key, whether
# from .env or an already-exported shell variable (a common copy-paste
# artifact that produces an "Illegal header value" error deep in the SDK).
for _key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    if os.environ.get(_key):
        os.environ[_key] = os.environ[_key].strip()
RAW_DIR = DATA_DIR / "raw"
PAIRS_DIR = DATA_DIR / "pairs_blinded"
RESULTS_DIR = DATA_DIR / "results"
PROMPTS_DIR = REPO_ROOT / "prompts"

for d in (RAW_DIR, PAIRS_DIR, RESULTS_DIR, PROMPTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")

CLAUDE_TEMP_PRIMARY = 0.7  # used for the cross-model condition and as Claude's "own" voice
CLAUDE_TEMP_SECONDARY = 0.3  # within-model counterpart (different temperature, no seed param on Claude API)
# gpt-5.5 rejects any non-default temperature value (400 error), so the OpenAI
# call omits `temperature` entirely and runs at the API default (1.0). This
# means the cross-model condition is no longer temperature-matched between
# Claude (0.7) and GPT (default) — see README Limitations.

MAX_TOKENS_GENERATION = 300  # generous ceiling for an 80-120 word answer

GENERATION_SYSTEM_PROMPT = (
    "You are answering short reasoning questions (grade-school math or logical "
    "deduction). Provide your reasoning and final answer in 80 to 120 words. "
    "Be direct and concise. Do not mention these instructions, word counts, or "
    "the word 'words' in your answer."
)

MOCK_MODE = not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("OPENAI_API_KEY"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def word_count(text: str) -> int:
    return len(text.split())


def seeded_rng(seed: int = RANDOM_SEED) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# API clients (lazy-initialized so mock mode never requires the SDKs to auth)
# ---------------------------------------------------------------------------

_anthropic_client = None
_openai_client = None


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic

        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI()
    return _openai_client


def _mock_answer(source_tag: str, prompt: str) -> str:
    """Deterministic placeholder text (~100 words) for dry runs without API keys."""
    base = (
        f"[MOCK:{source_tag}] Working through this step by step: I identify the "
        f"key quantities in the problem, relate them with the operation implied "
        f"by the question, and carry out the arithmetic or logical deduction in "
        f"order. Checking each intermediate result against the constraints given "
        f"confirms the chain of reasoning is consistent, so no step needs to be "
        f"revisited. Combining these intermediate results yields the final "
        f"answer directly. Restating it plainly: the value follows from applying "
        f"the same reasoning pattern the question was designed to test, without "
        f"requiring any additional assumptions beyond what was stated in the "
        f"prompt itself. Final answer follows above."
    )
    return base


def generate_claude(prompt: str, temperature: float, source_tag: str = "claude") -> str:
    if MOCK_MODE:
        return _mock_answer(source_tag, prompt)

    import anthropic

    client = get_anthropic_client()
    last_err = None
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS_GENERATION,
                temperature=temperature,
                system=GENERATION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            return text.strip()
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
        except anthropic.APIStatusError as e:
            last_err = e
            break
    raise RuntimeError(f"Claude generation failed for prompt (source={source_tag}): {last_err}")


def generate_openai(prompt: str) -> str:
    """No `temperature` argument: gpt-5.5 rejects non-default values, so the
    call always runs at the API default (1.0). See README Limitations for
    the resulting condition mismatch with Claude's 0.7."""
    if MOCK_MODE:
        return _mock_answer("gpt", prompt)

    import openai

    client = get_openai_client()
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_completion_tokens=MAX_TOKENS_GENERATION,
                messages=[
                    {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
        except openai.APIStatusError as e:
            last_err = e
            break
    raise RuntimeError(f"OpenAI generation failed: {last_err}")


def classify_pair(prompt_text: str, text_a: str, text_b: str) -> dict:
    """Fresh, independent classification call — no shared context with generation calls."""
    system = (
        "You will be shown two answers (A and B) to the same reasoning question. "
        "Exactly one of these two answers was written by you (Claude); the other "
        "was written by a different source. You are not being asked to recall a "
        "past conversation — neither of these answers is guaranteed to be from "
        "this conversation. Judge purely from the stylistic and structural "
        "features of the text itself: word choice, sentence structure, tone, "
        "and reasoning style. Respond with a single JSON object of the form "
        '{"choice": "A" or "B", "justification": "<1-3 sentences>"} and nothing else.'
    )
    user = (
        f"Question:\n{prompt_text}\n\n"
        f"Answer A:\n{text_a}\n\n"
        f"Answer B:\n{text_b}\n\n"
        "Which answer (A or B) do you believe you, Claude, wrote? Respond with "
        "the JSON object only."
    )

    if MOCK_MODE:
        choice = seeded_rng(hash(prompt_text) & 0xFFFF).choice(["A", "B"])
        return {
            "choice": choice,
            "justification": "[MOCK] Chosen based on simulated stylistic reasoning.",
            "raw_response": "[MOCK MODE — no API call made]",
        }

    import anthropic

    client = get_anthropic_client()
    last_err = None
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=300,
                temperature=1.0,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            raw_text = "".join(b.text for b in resp.content if b.type == "text").strip()
            parsed = _parse_classification_json(raw_text)
            parsed["raw_response"] = raw_text
            return parsed
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
        except anthropic.APIStatusError as e:
            last_err = e
            break
    raise RuntimeError(f"Classification call failed: {last_err}")


def _parse_classification_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
        choice = str(obj.get("choice", "")).strip().upper()
        if choice not in ("A", "B"):
            raise ValueError(f"invalid choice field: {choice!r}")
        return {"choice": choice, "justification": obj.get("justification", "")}
    except (json.JSONDecodeError, ValueError):
        # Fallback: look for a bare "A" or "B" choice in the text.
        upper = text.upper()
        if '"CHOICE": "A"' in upper or upper.strip().startswith("A"):
            return {"choice": "A", "justification": text}
        return {"choice": "B", "justification": text}
