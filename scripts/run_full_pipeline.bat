@echo off
cd /d %~dp0..
echo [%date% %time%] Starting full experiment pipeline...
python experiments/run_all.py --device auto 2>&1 | tee results/run_log.txt
echo [%date% %time%] Experiment done. Running analysis...
python scripts/analyze_and_optimize.py
python scripts/apply_round2_config.py
echo [%date% %time%] Round-2 config ready. DONE.
