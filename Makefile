.PHONY: setup run run-phase2 run-uci clean
setup:
	pip install -r requirements.txt
run:
	for s in 1_reproduction 2_performance_analysis 3_threshold_tuning 4_shap_analysis 5_lightgbm; do python pipeline/step$$s.py || exit 1; done
run-phase2:
	cd phase2 && for s in p2_step1_tuned_weighted_stack p2_step2_calibration p2_step3_bootstrap p2_step4_cost_threshold; do python $$s.py || exit 1; done
run-uci:
	cd phase2 && python p2_step9_multidataset.py && python p2_step10_protocol_audit.py
clean:
	rm -rf artifacts
