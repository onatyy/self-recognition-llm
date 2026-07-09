.PHONY: all install pipeline sample generate pairs classify classify_swapped analyze position_control debiased_analyze qualitative visualize clean

all: pipeline

install:
	pip install -r requirements.txt

pipeline:
	python run_pipeline.py

sample:
	python src/sample_prompts.py

generate:
	python src/stage1_generate.py

pairs:
	python src/stage2_pairs.py

classify:
	python src/stage3_classify.py

classify_swapped:
	python src/stage3b_classify_swapped.py

analyze:
	python src/stage4_analyze.py

position_control:
	python src/stage4b_position_control.py

debiased_analyze:
	python src/stage4c_debiased_analysis.py

qualitative:
	python src/stage5_qualitative.py

visualize:
	python src/stage6_visualize.py

clean:
	rm -f prompts/prompts.json data/raw/*.json data/pairs_blinded/*.json data/results/*.json data/results/*.png
