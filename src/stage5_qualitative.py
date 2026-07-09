"""Stage 5: Qualitative analysis of justifications.

Categorizes each justification text from Stage 3 into one of six fixed
categories using keyword matching (a lightweight heuristic appropriate for
a research script — not a trained classifier), then compares the category
distribution between correct and incorrect predictions.

Categories:
  - word_choice:        vocabulary, phrasing, specific word choices
  - sentence_structure:  syntax, sentence length/complexity, punctuation
  - tone:                formality, warmth, confidence, voice
  - reasoning_style:     how the argument/logic is structured or sequenced
  - quality_heuristic:   judged by perceived correctness/quality rather than style
  - vague_generic:       no specific stylistic feature named ("it just feels like mine")
"""

from collections import Counter, defaultdict

from utils import PAIRS_DIR, RESULTS_DIR, load_json, save_json

CATEGORY_KEYWORDS = {
    "word_choice": ["word choice", "vocabulary", "phrasing", "wording", "diction", "terminology", "word choose"],
    "sentence_structure": ["sentence structure", "syntax", "sentence length", "punctuation", "clause", "structure of the sentence", "grammar"],
    "tone": ["tone", "voice", "formal", "informal", "warmth", "confident", "conversational", "style of writing", "register"],
    "reasoning_style": ["reasoning", "logic", "approach", "step-by-step", "step by step", "argument", "explanation style", "how it explains", "structured the solution"],
    "quality_heuristic": ["correct", "accurate", "better", "quality", "more precise", "clearer", "more thorough", "more detailed", "well-reasoned", "polished"],
}


def categorize(justification: str) -> str:
    text = justification.lower()
    hits = Counter()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                hits[category] += 1
    if not hits:
        return "vague_generic"
    return hits.most_common(1)[0][0]


def main():
    results = load_json(RESULTS_DIR / "classification_results.json")
    ground_truth = {g["pair_id"]: g for g in load_json(PAIRS_DIR / "ground_truth.json")}

    categorized = []
    for r in results:
        gt = ground_truth[r["pair_id"]]
        is_correct = r["choice"] == gt["claude_side"]
        category = categorize(r["justification"])
        categorized.append(
            {
                "pair_id": r["pair_id"],
                "condition": r["condition"],
                "correct": is_correct,
                "category": category,
                "justification": r["justification"],
            }
        )

    by_correctness = defaultdict(Counter)
    for c in categorized:
        by_correctness["correct" if c["correct"] else "incorrect"][c["category"]] += 1

    summary = {
        "categorized_justifications": categorized,
        "category_counts_by_correctness": {
            k: dict(v) for k, v in by_correctness.items()
        },
        "overall_category_counts": dict(Counter(c["category"] for c in categorized)),
    }

    save_json(summary, RESULTS_DIR / "qualitative_analysis.json")

    print("=== Justification category distribution ===")
    print("Overall:", dict(Counter(c["category"] for c in categorized)))
    for k, v in by_correctness.items():
        print(f"{k}: {dict(v)}")
    print(f"\nWrote qualitative analysis -> {RESULTS_DIR / 'qualitative_analysis.json'}")


if __name__ == "__main__":
    main()
