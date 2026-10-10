.PHONY: setup run clean
setup:
	pip install -r requirements.txt
run:
	for s in 1_reproduction 2_performance_analysis 3_threshold_tuning 4_shap_analysis 5_lightgbm; do python pipeline/step$$s.py || exit 1; done
clean:
	rm -rf artifacts
