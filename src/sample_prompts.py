"""Sample the fixed prompt set: 15 GSM8K problems + 10 BIG-Bench-Hard logical
reasoning problems, filtered to ones short enough to plausibly be answered in
80-120 words. Writes prompts/prompts.json (reproducible via RANDOM_SEED).
"""

from utils import PROMPTS_DIR, RANDOM_SEED, save_json, seeded_rng

N_GSM8K = 15
N_BBH = 10
BBH_SUBTASKS = ["logical_deduction_three_objects", "causal_judgement"]

# Problem statements longer than this are dropped before sampling — a very
# long problem statement tends to require a long answer to address fully,
# which risks blowing the 80-120 word budget. This is a heuristic proxy;
# the true constraint (answer length) is enforced downstream by the
# generation system prompt and validated after generation.
MAX_PROBLEM_WORDS = 120


def _load_gsm8k():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    candidates = [
        {"prompt_id": f"gsm8k_{i}", "domain": "gsm8k", "question": row["question"]}
        for i, row in enumerate(ds)
        if len(row["question"].split()) <= MAX_PROBLEM_WORDS
    ]
    rng = seeded_rng(RANDOM_SEED)
    rng.shuffle(candidates)
    return candidates[:N_GSM8K]


def _load_bbh():
    from datasets import load_dataset

    candidates = []
    for subtask in BBH_SUBTASKS:
        ds = load_dataset("lukaemon/bbh", subtask, split="test")
        for i, row in enumerate(ds):
            if len(row["input"].split()) <= MAX_PROBLEM_WORDS:
                candidates.append(
                    {
                        "prompt_id": f"bbh_{subtask}_{i}",
                        "domain": f"bbh_{subtask}",
                        "question": row["input"],
                    }
                )
    rng = seeded_rng(RANDOM_SEED + 1)
    rng.shuffle(candidates)
    return candidates[:N_BBH]


def main():
    prompts = _load_gsm8k() + _load_bbh()
    out_path = PROMPTS_DIR / "prompts.json"
    save_json(prompts, out_path)
    print(f"Sampled {len(prompts)} prompts ({N_GSM8K} GSM8K + {N_BBH} BBH) -> {out_path}")


if __name__ == "__main__":
    main()
