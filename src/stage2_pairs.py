"""Stage 2: Pair construction and randomization.

For each prompt, build:
  - one cross-model pair:  claude_t0.7  vs  gpt_default
  - one within-model pair: claude_t0.7  vs  claude_t0.3

For each pair, randomly assign which text is shown as "A" and which as "B"
(fixed seed -> reproducible). The blinded classification input (what Stage 3
sees) and the ground-truth labels (which side is actually Claude's) are
written to two separate files so it's easy to verify no leakage occurred.
"""

from utils import PAIRS_DIR, RAW_DIR, RANDOM_SEED, load_json, save_json, seeded_rng

CROSS_MODEL_SOURCES = ("claude_t0.7", "gpt_default")
WITHIN_MODEL_SOURCES = ("claude_t0.7", "claude_t0.3")


def _index_by_prompt(generations):
    by_prompt = {}
    for g in generations:
        by_prompt.setdefault(g["prompt_id"], {})[g["source_tag"]] = g
    return by_prompt


def _build_pair(pair_id, condition, prompt_id, question, gen_claude, gen_other, rng):
    claude_side = rng.choice(["A", "B"])
    if claude_side == "A":
        text_a, text_b = gen_claude["raw_text"], gen_other["raw_text"]
        source_a, source_b = gen_claude["source_tag"], gen_other["source_tag"]
    else:
        text_a, text_b = gen_other["raw_text"], gen_claude["raw_text"]
        source_a, source_b = gen_other["source_tag"], gen_claude["source_tag"]

    blinded = {
        "pair_id": pair_id,
        "condition": condition,
        "prompt_id": prompt_id,
        "prompt_text": question,
        "text_a": text_a,
        "text_b": text_b,
    }
    ground_truth = {
        "pair_id": pair_id,
        "condition": condition,
        "prompt_id": prompt_id,
        "claude_side": claude_side,
        "source_a": source_a,
        "source_b": source_b,
    }
    return blinded, ground_truth


def main():
    generations = load_json(RAW_DIR / "generations.json")
    by_prompt = _index_by_prompt(generations)
    rng = seeded_rng(RANDOM_SEED)

    blinded_pairs = []
    ground_truth = []

    for prompt_id, sources in by_prompt.items():
        question = sources[CROSS_MODEL_SOURCES[0]]["question"]

        b, g = _build_pair(
            pair_id=f"{prompt_id}_cross",
            condition="cross_model",
            prompt_id=prompt_id,
            question=question,
            gen_claude=sources[CROSS_MODEL_SOURCES[0]],
            gen_other=sources[CROSS_MODEL_SOURCES[1]],
            rng=rng,
        )
        blinded_pairs.append(b)
        ground_truth.append(g)

        b, g = _build_pair(
            pair_id=f"{prompt_id}_within",
            condition="within_model",
            prompt_id=prompt_id,
            question=question,
            gen_claude=sources[WITHIN_MODEL_SOURCES[0]],
            gen_other=sources[WITHIN_MODEL_SOURCES[1]],
            rng=rng,
        )
        blinded_pairs.append(b)
        ground_truth.append(g)

    save_json(blinded_pairs, PAIRS_DIR / "classification_input.json")
    save_json(ground_truth, PAIRS_DIR / "ground_truth.json")
    print(f"Wrote {len(blinded_pairs)} blinded pairs -> {PAIRS_DIR / 'classification_input.json'}")
    print(f"Wrote {len(ground_truth)} ground-truth labels -> {PAIRS_DIR / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
