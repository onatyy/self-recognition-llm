"""Stage 4: Statistical analysis.

Joins the blinded classification results with the ground-truth labels
(the only point in the pipeline where the two are combined), then:
  - computes accuracy per condition (cross_model, within_model)
  - runs a two-sided binomial test against chance (0.5) for each condition
  - compares accuracy between the two conditions (Fisher's exact test)
  - runs a chi-square goodness-of-fit test for positional bias (A vs B)
"""

from scipy.stats import binomtest, chi2_contingency, fisher_exact, chisquare

from utils import PAIRS_DIR, RESULTS_DIR, load_json, save_json

CONDITIONS = ["cross_model", "within_model"]


def _binom_summary(n_correct: int, n_total: int) -> dict:
    result = binomtest(n_correct, n_total, p=0.5, alternative="two-sided")
    ci = result.proportion_ci(confidence_level=0.95)
    return {
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": n_correct / n_total if n_total else None,
        "p_value": result.pvalue,
        "ci_95_low": ci.low,
        "ci_95_high": ci.high,
    }


def main():
    results = load_json(RESULTS_DIR / "classification_results.json")
    ground_truth = {g["pair_id"]: g for g in load_json(PAIRS_DIR / "ground_truth.json")}

    by_condition = {c: {"correct": 0, "total": 0, "choices": []} for c in CONDITIONS}

    for r in results:
        gt = ground_truth[r["pair_id"]]
        is_correct = r["choice"] == gt["claude_side"]
        cond = r["condition"]
        by_condition[cond]["total"] += 1
        by_condition[cond]["correct"] += int(is_correct)
        by_condition[cond]["choices"].append(r["choice"])

    condition_stats = {c: _binom_summary(v["correct"], v["total"]) for c, v in by_condition.items()}

    # Compare accuracy between conditions: 2x2 contingency table (correct/incorrect x condition)
    table = [
        [by_condition["cross_model"]["correct"], by_condition["cross_model"]["total"] - by_condition["cross_model"]["correct"]],
        [by_condition["within_model"]["correct"], by_condition["within_model"]["total"] - by_condition["within_model"]["correct"]],
    ]
    _, fisher_p = fisher_exact(table)

    # Positional bias: does Claude systematically favor "A" or "B" regardless of content?
    all_choices = [r["choice"] for r in results]
    n_a = all_choices.count("A")
    n_b = all_choices.count("B")
    chi2_stat, chi2_p = chisquare([n_a, n_b], f_exp=[len(all_choices) / 2, len(all_choices) / 2])

    summary = {
        "condition_stats": condition_stats,
        "cross_model_vs_within_model": {
            "contingency_table": table,
            "fisher_exact_p_value": fisher_p,
        },
        "positional_bias": {
            "n_A": n_a,
            "n_B": n_b,
            "chi2_statistic": chi2_stat,
            "p_value": chi2_p,
        },
    }

    save_json(summary, RESULTS_DIR / "statistics.json")

    print("=== Accuracy by condition (binomial test vs. chance = 0.5) ===")
    for c, s in condition_stats.items():
        print(
            f"{c}: {s['n_correct']}/{s['n_total']} = {s['accuracy']:.3f} "
            f"(95% CI [{s['ci_95_low']:.3f}, {s['ci_95_high']:.3f}]), p={s['p_value']:.4f}"
        )
    print(f"\nCross-model vs. within-model accuracy (Fisher's exact test): p={fisher_p:.4f}")
    print(f"\nPositional bias (A={n_a}, B={n_b}): chi2={chi2_stat:.3f}, p={chi2_p:.4f}")
    print(f"\nWrote full statistics -> {RESULTS_DIR / 'statistics.json'}")


if __name__ == "__main__":
    main()
