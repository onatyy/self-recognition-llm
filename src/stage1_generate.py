"""Stage 1: Stimulus generation.

For each sampled prompt, generate three answers:
  - Claude at temperature 0.7      (source: claude_t0.7)
  - GPT (OpenAI) at API default    (source: gpt_default; gpt-5.5 rejects any
    non-default temperature, so this is not temperature-matched to Claude's
    0.7 — see README Limitations)
  - Claude at temperature 0.3      (source: claude_t0.3, within-model counterpart)

All generations share the same system instruction and an 80-120 word target.
Every generated answer is stored with full metadata (prompt id, source model,
temperature, timestamp, raw text) in data/raw/generations.json.
"""

from utils import (
    CLAUDE_MODEL,
    CLAUDE_TEMP_PRIMARY,
    CLAUDE_TEMP_SECONDARY,
    MOCK_MODE,
    OPENAI_MODEL,
    PROMPTS_DIR,
    RAW_DIR,
    generate_claude,
    generate_openai,
    load_json,
    now_iso,
    save_json,
    word_count,
)


def main():
    prompts = load_json(PROMPTS_DIR / "prompts.json")
    if MOCK_MODE:
        print("MOCK MODE: no ANTHROPIC_API_KEY/OPENAI_API_KEY found — generating placeholder text.")

    generations = []
    for p in prompts:
        pid, question = p["prompt_id"], p["question"]
        print(f"Generating for {pid}...")

        claude_primary = generate_claude(question, CLAUDE_TEMP_PRIMARY, source_tag="claude_t0.7")
        generations.append(
            {
                "prompt_id": pid,
                "domain": p["domain"],
                "question": question,
                "source_model": CLAUDE_MODEL,
                "source_tag": "claude_t0.7",
                "temperature": CLAUDE_TEMP_PRIMARY,
                "timestamp": now_iso(),
                "raw_text": claude_primary,
                "word_count": word_count(claude_primary),
            }
        )

        gpt_answer = generate_openai(question)
        generations.append(
            {
                "prompt_id": pid,
                "domain": p["domain"],
                "question": question,
                "source_model": OPENAI_MODEL,
                "source_tag": "gpt_default",
                "temperature": None,  # gpt-5.5 rejects non-default temperature; runs at API default (1.0)
                "timestamp": now_iso(),
                "raw_text": gpt_answer,
                "word_count": word_count(gpt_answer),
            }
        )

        claude_secondary = generate_claude(question, CLAUDE_TEMP_SECONDARY, source_tag="claude_t0.3")
        generations.append(
            {
                "prompt_id": pid,
                "domain": p["domain"],
                "question": question,
                "source_model": CLAUDE_MODEL,
                "source_tag": "claude_t0.3",
                "temperature": CLAUDE_TEMP_SECONDARY,
                "timestamp": now_iso(),
                "raw_text": claude_secondary,
                "word_count": word_count(claude_secondary),
            }
        )

    out_path = RAW_DIR / "generations.json"
    save_json(generations, out_path)
    print(f"Wrote {len(generations)} generations -> {out_path}")


if __name__ == "__main__":
    main()
