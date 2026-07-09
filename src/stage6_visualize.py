"""Stage 6: Visualization.

Primary chart: position-debiased accuracy per condition (Stage 4c), the
headline result, with a horizontal reference line at chance level (0.5) and
error bars from the debiased 95% confidence interval.

Secondary chart: single-orientation accuracy per condition (Stage 4), kept
for the documented methodological finding further down in the README.
"""

import matplotlib.pyplot as plt

from utils import RESULTS_DIR, load_json

CONDITION_LABELS = {"cross_model": "Cross-model\n(Claude vs. GPT)", "within_model": "Within-model\n(Claude vs. Claude)"}
BAR_COLOR = "#3B6FA0"
CHANCE_COLOR = "#888888"


def _plot(condition_stats: dict, n_key: str, n_label: str, title: str, out_path) -> None:
    conditions = list(CONDITION_LABELS.keys())
    labels = [CONDITION_LABELS[c] for c in conditions]
    accuracies = [condition_stats[c]["accuracy"] for c in conditions]
    err_low = [condition_stats[c]["accuracy"] - condition_stats[c]["ci_95_low"] for c in conditions]
    err_high = [condition_stats[c]["ci_95_high"] - condition_stats[c]["accuracy"] for c in conditions]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = range(len(conditions))
    ax.bar(x, accuracies, yerr=[err_low, err_high], capsize=8, color=BAR_COLOR, width=0.5, zorder=3)
    ax.axhline(0.5, color=CHANCE_COLOR, linestyle="--", linewidth=1.5, label="Chance (0.5)", zorder=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for xi, acc, s in zip(x, accuracies, [condition_stats[c] for c in conditions]):
        ax.text(xi, acc + err_high[list(x).index(xi)] + 0.03, f"{acc:.2f}\n({n_label}={s[n_key]})", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote chart -> {out_path}")


def main():
    debiased = load_json(RESULTS_DIR / "debiased_statistics.json")
    _plot(
        debiased["condition_stats"],
        n_key="n_pairs",
        n_label="n",
        title="Claude's self-recognition accuracy by condition\n(position-debiased, primary result)",
        out_path=RESULTS_DIR / "accuracy_chart.png",
    )

    single_orientation = load_json(RESULTS_DIR / "statistics.json")
    _plot(
        single_orientation["condition_stats"],
        n_key="n_total",
        n_label="n",
        title="Claude's self-recognition accuracy by condition\n(single orientation, methodological finding)",
        out_path=RESULTS_DIR / "accuracy_chart_single_orientation.png",
    )


if __name__ == "__main__":
    main()
