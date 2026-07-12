# Makefile for elayra-research
# Usage:  make help
# Requires: Python ≥3.10, pip-installed package (pip install -e ".[dev]")

PY      := .venv/Scripts/python.exe
PYFLAGS := -O
SCRIPT   = scripts
RESULT   = results
META_FILE= $(RESULT)/experiment_meta.json

.PHONY: help install install-dev test pipeline layerwise \
        control control-short control-long control-shuffled \
        init-analysis bootstrap l1 checkpoint heads modern-llms mlp \
        mlp-embedding layerwise-modern extend \
        results all clean clean-results

help:
	@echo "Targets:"
	@echo "  make install        — install editable package"
	@echo "  make install-dev    — also install pytest etc."
	@echo "  make test           — run tests/"
	@echo "  make all            — pipeline + controls + analyses"
	@echo "  make results        — regenerate results/"
	@echo "  make clean          — remove .pyc / cache"
	@echo "  make clean-results  — remove results/"

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/ -v

pipeline:
	$(PY) $(SCRIPT)/run_pipeline.py

layerwise:
	$(PY) $(SCRIPT)/run_layerwise.py

control-short:
	$(PY) $(SCRIPT)/control_short.py

control-long:
	$(PY) $(SCRIPT)/control_long.py

control-shuffled:
	$(PY) $(SCRIPT)/control_shuffled.py

controls: control-short control-long control-shuffled

init-analysis:
	$(PY) $(SCRIPT)/init_analysis.py

bootstrap:
	$(PY) $(SCRIPT)/bootstrap_analysis.py

l1:
	$(PY) $(SCRIPT)/l1_regularization_test.py

checkpoint:
	$(PY) $(SCRIPT)/checkpoint_analysis.py

heads:
	$(PY) $(SCRIPT)/head_level_analysis.py

modern-llms:
	$(PY) $(SCRIPT)/modern_llms_ext.py

mlp:
	$(PY) $(SCRIPT)/mlp_analysis.py

mlp-embedding:
	$(PY) $(SCRIPT)/mlp_embedding_analysis.py

layerwise-modern:
	$(PY) $(SCRIPT)/run_layerwise_modern.py

extend: mlp-embedding heads control-short
	@echo "Incremental extension run done (MLP/embedding + fix confirmation)."

results: pipeline layerwise controls init-analysis bootstrap l1 checkpoint heads modern-llms mlp
	@echo "All results regenerated."

all: install results test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true

clean-results:
	rm -f $(RESULT)/*.json $(RESULT)/*.png $(RESULT)/*.svg
	#rm -f hidden_results/*.json hidden_results/*.png
