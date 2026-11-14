.PHONY: setup run run-phase2 run-uci review2 bundle serve test clean
setup:
	pip install -r requirements.txt
run:
	for s in 1_reproduction 2_performance_analysis 3_threshold_tuning 4_shap_analysis 5_lightgbm; do python pipeline/step$$s.py || exit 1; done
run-phase2:
	cd phase2 && for s in p2_step1_tuned_weighted_stack p2_step2_calibration p2_step3_bootstrap p2_step4_cost_threshold; do python $$s.py || exit 1; done
run-uci:
	cd phase2 && python p2_step9_multidataset.py && python p2_step10_protocol_audit.py
review2:
	cd phase2 && for s in p2_step11_stack_protocol_audit p2_step12_decision_policy p2_step13_build_bundle p2_step14_reason_codes p2_step15_monitoring p2_step16_scorecard_baseline; do python $$s.py || exit 1; done
bundle:
	cd phase2 && python p2_step13_build_bundle.py
serve:
	uvicorn creditsense.api:app --reload
test:
	python -m pytest -q tests
clean:
	rm -rf artifacts
