"""Stage 3: Classification task (the actual experiment).

For each blinded pair, open a fresh, independent API call to Claude (no
shared conversation context with Stage 1 generations, and no context from
any other pair) and ask it to identify which text it believes it wrote,
with a brief justification. Only classification_input.json (the blinded
file) is read here — ground_truth.json is never touched by this script.
"""

from utils import PAIRS_DIR, RESULTS_DIR, classify_pair, load_json, save_json


def main():
    pairs = load_json(PAIRS_DIR / "classification_input.json")

    results = []
    for pair in pairs:
        print(f"Classifying {pair['pair_id']}...")
        outcome = classify_pair(pair["prompt_text"], pair["text_a"], pair["text_b"])
        results.append(
            {
                "pair_id": pair["pair_id"],
                "condition": pair["condition"],
                "choice": outcome["choice"],
                "justification": outcome["justification"],
                "raw_response": outcome["raw_response"],
            }
        )

    out_path = RESULTS_DIR / "classification_results.json"
    save_json(results, out_path)
    print(f"Wrote {len(results)} classification results -> {out_path}")


if __name__ == "__main__":
    main()
