.PHONY: help setup test notebooks check-notebooks report zip clean

help: ## show this help
	@grep -E '^[a-z0-9%-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-16s %s\n", $$1, $$2}'

setup: ## install dependencies with uv
	uv sync --all-groups

test: ## run the CPU test suite
	uv run pytest -q

notebooks: ## generate the notebooks into build/notebooks
	uv run python scripts/build_notebooks.py

check-notebooks: ## fail if an executed notebook differs from a fresh build
	uv run python scripts/build_notebooks.py --check

colab-%: notebooks ## run one activity on a Colab T4 and collect results (make colab-a3)
	uv run python scripts/colab_run.py $*

report: ## build the PDF report
	uv run python scripts/build_report.py

zip: report ## build the delivery archive
	mkdir -p entrega
	rm -f entrega/anderson_correa_computer-vision_pd.zip
	zip -j entrega/anderson_correa_computer-vision_pd.zip notebooks/*.ipynb report/anderson_correa_deep-learning-and-vision_computer-vision.pdf

clean: ## remove generated files
	rm -rf build entrega
