#!/usr/bin/env python3
"""End-to-end pipeline entry point.

Usage:
    python run_pipeline.py            # run all stages
    python run_pipeline.py --from 3   # resume from stage 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import sample_prompts
import stage1_generate
import stage2_pairs
import stage3_classify
import stage3b_classify_swapped
import stage4_analyze
import stage4b_position_control
import stage4c_debiased_analysis
import stage5_qualitative
import stage6_visualize

STAGES = [
    ("0. Sample prompts", sample_prompts.main),
    ("1. Generate stimuli", stage1_generate.main),
    ("2. Build & randomize pairs", stage2_pairs.main),
    ("3. Classify pairs", stage3_classify.main),
    ("3b. Classify pairs (swapped orientation)", stage3b_classify_swapped.main),
    ("4. Statistical analysis", stage4_analyze.main),
    ("4b. Position-controlled analysis", stage4b_position_control.main),
    ("4c. Position-debiased analysis (primary result)", stage4c_debiased_analysis.main),
    ("5. Qualitative analysis", stage5_qualitative.main),
    ("6. Visualization", stage6_visualize.main),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_stage", type=int, default=0, help="Stage index to resume from (0-9)")
    args = parser.parse_args()

    for i, (name, fn) in enumerate(STAGES):
        if i < args.from_stage:
            continue
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        fn()


if __name__ == "__main__":
    main()
