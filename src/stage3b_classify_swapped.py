"""Stage 3b: Swapped-orientation classification (position debiasing).

Stage 4b showed that Claude's raw accuracy is confounded by a strong
positional bias (it favors "A" almost regardless of content). Rather than
just documenting that confound, this stage collects the data needed to
correct for it directly: for every pair already built in Stage 2, the A and
B texts are swapped and a fresh, independent classification call is made
for the swapped presentation, exactly as in Stage 3 (no shared conversation
context with Stage 1, Stage 3, or any other Stage 3b call).

Every original pair is therefore judged in both orderings. Stage 4c averages
correctness across the two orderings per pair to get a position-debiased
accuracy estimate: a pair only counts as "recognized" to the extent Claude
gets it right regardless of which side its real text was shown on.

Reads only classification_input.json (the blinded file) — same blinding
guarantee as Stage 3. Writes to a separate results file so the original,
single-orientation results (Stage 3/4/4b) are never overwritten.
"""

from utils import PAIRS_DIR, RESULTS_DIR, classify_pair, load_json, save_json


def _swap(pair: dict) -> dict:
    return {
        **pair,
        "text_a": pair["text_b"],
        "text_b": pair["text_a"],
    }


def main():
    pairs = load_json(PAIRS_DIR / "classification_input.json")

    results = []
    for pair in pairs:
        print(f"Classifying (swapped) {pair['pair_id']}...")
        swapped = _swap(pair)
        outcome = classify_pair(swapped["prompt_text"], swapped["text_a"], swapped["text_b"])
        results.append(
            {
                "pair_id": pair["pair_id"],
                "condition": pair["condition"],
                "orientation": "swapped",
                "choice": outcome["choice"],
                "justification": outcome["justification"],
                "raw_response": outcome["raw_response"],
            }
        )

    out_path = RESULTS_DIR / "classification_results_swapped.json"
    save_json(results, out_path)
    print(f"Wrote {len(results)} swapped-orientation classification results -> {out_path}")


if __name__ == "__main__":
    main()
