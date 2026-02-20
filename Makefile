PYTHON ?= python
BASE_URL ?= http://localhost:8000
QUERY_PATH ?= /v1/query
EVAL_FILE ?= data/eval_queries.json
RUNS_DIR ?= runs
K ?= 10

BASELINE_RUN ?= $(RUNS_DIR)/baseline_run.json
BASELINE_SCORE ?= $(RUNS_DIR)/baseline_score.json
CANDIDATE_RUN ?= $(RUNS_DIR)/candidate_run.json
CANDIDATE_SCORE ?= $(RUNS_DIR)/candidate_score.json
COMPARE_OUT ?= $(RUNS_DIR)/compare.json

.PHONY: help eval-prepare eval-baseline-run eval-baseline-score eval-baseline eval-candidate-run eval-candidate-score eval-candidate eval-compare eval-all

help:
	@echo "Targets:"
	@echo "  make eval-prepare        # Create data/eval_queries.json from template if missing"
	@echo "  make eval-baseline       # Run + score baseline"
	@echo "  make eval-candidate      # Run + score candidate"
	@echo "  make eval-compare        # Compare baseline_score vs candidate_score"
	@echo "  make eval-all            # baseline + candidate + compare"
	@echo ""
	@echo "Configurable vars:"
	@echo "  BASE_URL=$(BASE_URL)"
	@echo "  EVAL_FILE=$(EVAL_FILE)"
	@echo "  RUNS_DIR=$(RUNS_DIR)"
	@echo "  K=$(K)"

eval-prepare:
	@mkdir -p $(RUNS_DIR)
	@if [ ! -f "$(EVAL_FILE)" ]; then \
		cp data/eval_queries.template.json $(EVAL_FILE); \
		echo "Created $(EVAL_FILE) from template. Fill relevant_ids/relevance_by_id before scoring."; \
	else \
		echo "$(EVAL_FILE) already exists."; \
	fi

eval-baseline-run: eval-prepare
	$(PYTHON) scripts/eval_retrieval.py run \
		--eval-file $(EVAL_FILE) \
		--out $(BASELINE_RUN) \
		--base-url $(BASE_URL) \
		--query-path $(QUERY_PATH) \
		--run-name baseline

eval-baseline-score:
	$(PYTHON) scripts/eval_retrieval.py score \
		--eval-file $(EVAL_FILE) \
		--run-file $(BASELINE_RUN) \
		--out $(BASELINE_SCORE) \
		--k $(K)

eval-baseline: eval-baseline-run eval-baseline-score

eval-candidate-run: eval-prepare
	$(PYTHON) scripts/eval_retrieval.py run \
		--eval-file $(EVAL_FILE) \
		--out $(CANDIDATE_RUN) \
		--base-url $(BASE_URL) \
		--query-path $(QUERY_PATH) \
		--run-name candidate

eval-candidate-score:
	$(PYTHON) scripts/eval_retrieval.py score \
		--eval-file $(EVAL_FILE) \
		--run-file $(CANDIDATE_RUN) \
		--out $(CANDIDATE_SCORE) \
		--k $(K)

eval-candidate: eval-candidate-run eval-candidate-score

eval-compare:
	$(PYTHON) scripts/eval_retrieval.py compare \
		--baseline-score $(BASELINE_SCORE) \
		--candidate-score $(CANDIDATE_SCORE) \
		--out $(COMPARE_OUT)

eval-all: eval-baseline eval-candidate eval-compare
