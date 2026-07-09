"""Stage 4b: Position-controlled analysis.

Stage 4 established a strong positional bias (Claude systematically favors
one side of the A/B choice regardless of content). If Claude's real text is
disproportionately assigned to the favored side, the raw accuracy numbers in
Stage 4 could be an artifact of that bias rather than genuine stylistic
self-recognition.

This stage cross-tabulates the position of Claude's actual text (A or B,
from ground truth) against Claude's chosen answer (A or B), separately for
each condition, and recomputes accuracy conditioned on position: accuracy
among pairs where Claude's real text was A, versus accuracy among pairs
where it was B. If accuracy collapses toward chance (or below) when Claude's
real text was on the non-favored side, that is direct evidence the raw
accuracy figure is confounded by positional bias rather than reflecting
self-recognition.
"""

from collections import Counter

from scipy.stats import binomtest

from utils import PAIRS_DIR, RESULTS_DIR, load_json, save_json

CONDITIONS = ["cross_model", "within_model"]
POSITIONS = ["A", "B"]


def _binom_summary(n_correct: int, n_total: int) -> dict:
    if n_total == 0:
        return {
            "n_correct": 0,
            "n_total": 0,
            "accuracy": None,
            "p_value": None,
            "ci_95_low": None,
            "ci_95_high": None,
            "note": "no pairs in this subgroup",
        }
    result = binomtest(n_correct, n_total, p=0.5, alternative="two-sided")
    ci = result.proportion_ci(confidence_level=0.95)
    summary = {
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": n_correct / n_total,
        "p_value": result.pvalue,
        "ci_95_low": ci.low,
        "ci_95_high": ci.high,
    }
    if n_total < 10:
        summary["note"] = "small subgroup (n < 10); test is underpowered, treat p-value/CI as indicative only"
    return summary


def main():
    ground_truth = {g["pair_id"]: g for g in load_json(PAIRS_DIR / "ground_truth.json")}
    results = load_json(RESULTS_DIR / "classification_results.json")

    joined = []
    for r in results:
        gt = ground_truth[r["pair_id"]]
        joined.append(
            {
                "pair_id": r["pair_id"],
                "condition": r["condition"],
                "claude_side": gt["claude_side"],  # A or B: where Claude's real text actually was
                "choice": r["choice"],  # A or B: what Claude picked
                "correct": r["choice"] == gt["claude_side"],
            }
        )

    output = {}
    for condition in CONDITIONS:
        rows = [j for j in joined if j["condition"] == condition]

        # 2x2 cross-tab: rows = actual position of Claude's text, columns = Claude's choice
        cross_tab = {
            pos: Counter(j["choice"] for j in rows if j["claude_side"] == pos) for pos in POSITIONS
        }
        cross_tab_table = {
            pos: {"chose_A": cross_tab[pos].get("A", 0), "chose_B": cross_tab[pos].get("B", 0)}
            for pos in POSITIONS
        }

        # Accuracy conditioned on where Claude's real text actually was.
        accuracy_by_position = {}
        for pos in POSITIONS:
            subgroup = [j for j in rows if j["claude_side"] == pos]
            n_correct = sum(j["correct"] for j in subgroup)
            accuracy_by_position[pos] = _binom_summary(n_correct, len(subgroup))

        output[condition] = {
            "cross_tab_claude_side_vs_choice": cross_tab_table,
            "accuracy_when_claude_side_is_A": accuracy_by_position["A"],
            "accuracy_when_claude_side_is_B": accuracy_by_position["B"],
        }

    save_json(output, RESULTS_DIR / "position_control.json")

    print("=== Position-controlled accuracy (accuracy conditioned on where Claude's real text was) ===")
    for condition in CONDITIONS:
        c = output[condition]
        print(f"\n{condition}:")
        print(f"  cross-tab (claude_side -> choice): {c['cross_tab_claude_side_vs_choice']}")
        for pos in POSITIONS:
            s = c[f"accuracy_when_claude_side_is_{pos}"]
            if s["n_total"] == 0:
                print(f"  claude_side={pos}: no pairs")
                continue
            acc = s["accuracy"]
            p = s["p_value"]
            note = f" ({s['note']})" if "note" in s else ""
            print(f"  claude_side={pos}: {s['n_correct']}/{s['n_total']} = {acc:.3f}, p={p:.4f}{note}")

    print(f"\nWrote position-controlled analysis -> {RESULTS_DIR / 'position_control.json'}")


if __name__ == "__main__":
    main()
