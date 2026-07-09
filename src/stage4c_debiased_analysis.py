"""Stage 4c: Position-debiased accuracy (the primary reported result).

Combines the original-orientation classifications (Stage 3) with the
swapped-orientation classifications (Stage 3b) to compute, per pair, a
position-debiased score: the average of correctness in the original
presentation and correctness in the swapped presentation. A pair scores
1.0 only if Claude identifies its own text correctly regardless of which
side ("A" or "B") that text was shown on; 0.5 if it gets exactly one of
the two orientations right (the position-bias-driven outcome); 0.0 if it
gets both wrong.

Per condition, pairs are pooled into a single binomial test: all
correct/incorrect judgments across both orientations (2 x n_pairs trials)
are treated as the sample for a two-sided binomial test against chance
(0.5), giving accuracy, a p-value, and a 95% CI for the debiased measure.
This pooling assumes the two orientations of a pair behave, for testing
purposes, as approximately independent trials -- each is a fresh API call
with no shared context, but both judge the same underlying texts, so this
is a simplifying assumption, not a strict independence guarantee. It is
noted explicitly here and in the README.

The original single-orientation results (Stage 4) and the position-bias
diagnosis (Stage 4b) are left untouched as a documented methodological
finding; this stage's output is the primary reported result.
"""

from scipy.stats import binomtest

from utils import PAIRS_DIR, RESULTS_DIR, load_json, save_json

CONDITIONS = ["cross_model", "within_model"]
FLIP = {"A": "B", "B": "A"}


def _binom_summary(n_correct: int, n_total: int) -> dict:
    result = binomtest(n_correct, n_total, p=0.5, alternative="two-sided")
    ci = result.proportion_ci(confidence_level=0.95)
    return {
        "n_correct_judgments": n_correct,
        "n_total_judgments": n_total,
        "accuracy": n_correct / n_total,
        "p_value": result.pvalue,
        "ci_95_low": ci.low,
        "ci_95_high": ci.high,
    }


def main():
    ground_truth = {g["pair_id"]: g for g in load_json(PAIRS_DIR / "ground_truth.json")}
    original = {r["pair_id"]: r for r in load_json(RESULTS_DIR / "classification_results.json")}
    swapped = {r["pair_id"]: r for r in load_json(RESULTS_DIR / "classification_results_swapped.json")}

    per_pair = []
    for pair_id, gt in ground_truth.items():
        orig = original[pair_id]
        swap = swapped[pair_id]

        claude_side_original = gt["claude_side"]
        claude_side_swapped = FLIP[claude_side_original]

        correct_original = orig["choice"] == claude_side_original
        correct_swapped = swap["choice"] == claude_side_swapped
        debiased_score = (int(correct_original) + int(correct_swapped)) / 2

        per_pair.append(
            {
                "pair_id": pair_id,
                "condition": gt["condition"],
                "claude_side_original": claude_side_original,
                "choice_original": orig["choice"],
                "correct_original": correct_original,
                "claude_side_swapped": claude_side_swapped,
                "choice_swapped": swap["choice"],
                "correct_swapped": correct_swapped,
                "debiased_score": debiased_score,
            }
        )

    condition_stats = {}
    for condition in CONDITIONS:
        rows = [p for p in per_pair if p["condition"] == condition]
        n_correct_judgments = sum(int(p["correct_original"]) + int(p["correct_swapped"]) for p in rows)
        n_total_judgments = 2 * len(rows)

        score_distribution = {
            "1.0_both_orientations_correct": sum(1 for p in rows if p["debiased_score"] == 1.0),
            "0.5_one_orientation_correct": sum(1 for p in rows if p["debiased_score"] == 0.5),
            "0.0_both_orientations_incorrect": sum(1 for p in rows if p["debiased_score"] == 0.0),
        }

        stats = _binom_summary(n_correct_judgments, n_total_judgments)
        stats["n_pairs"] = len(rows)
        stats["mean_debiased_score_per_pair"] = sum(p["debiased_score"] for p in rows) / len(rows) if rows else None
        stats["score_distribution"] = score_distribution
        condition_stats[condition] = stats

    # Diagnostic: does the swapped-orientation batch reproduce the same
    # positional bias seen in Stage 4 (i.e. is the bias tied to presentation
    # position, not content)? A near-identical "chose A" skew here confirms
    # averaging the two orientations is the right fix.
    swapped_choices = [r["choice"] for r in swapped.values()]
    swapped_position_bias = {
        "n_A": swapped_choices.count("A"),
        "n_B": swapped_choices.count("B"),
    }

    output = {
        "condition_stats": condition_stats,
        "per_pair": per_pair,
        "swapped_orientation_position_bias_diagnostic": swapped_position_bias,
        "note": (
            "Per-condition binomial tests pool both orientations of every pair "
            "(2 x n_pairs judgments) as approximately independent trials. This "
            "is a simplifying assumption: each judgment is a fresh, independent "
            "API call, but both orientations of a pair judge the same underlying "
            "texts."
        ),
    }

    save_json(output, RESULTS_DIR / "debiased_statistics.json")

    print("=== Position-debiased accuracy (primary result): pooled original + swapped judgments ===")
    for condition, stats in condition_stats.items():
        print(
            f"{condition}: {stats['n_correct_judgments']}/{stats['n_total_judgments']} judgments correct "
            f"= {stats['accuracy']:.3f} (95% CI [{stats['ci_95_low']:.3f}, {stats['ci_95_high']:.3f}]), "
            f"p={stats['p_value']:.4f}, n_pairs={stats['n_pairs']}, "
            f"score distribution={stats['score_distribution']}"
        )
    print(
        f"\nSwapped-orientation choice distribution (diagnostic): "
        f"A={swapped_position_bias['n_A']}, B={swapped_position_bias['n_B']}"
    )
    print(f"\nWrote debiased statistics -> {RESULTS_DIR / 'debiased_statistics.json'}")


if __name__ == "__main__":
    main()
