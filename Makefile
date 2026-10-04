.PHONY: help setup test notebooks check-notebooks colab-all report zip clean

help: ## show this help
	@grep -E '^[a-z0-9%-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-16s %s\n", $$1, $$2}'

setup: ## install dependencies with uv
	uv sync --all-groups

test: ## run the CPU test suite
	uv run --all-groups pytest -q

notebooks: ## generate the notebooks into build/notebooks
	uv run python scripts/build_notebooks.py

check-notebooks: ## fail if an executed notebook differs from a fresh build
	uv run python scripts/build_notebooks.py --check

colab-all: ## run a3, a2, a1 and a4 on Colab in sequence, stopping at the first failure
	@for activity in a3 a2 a1 a4; do $(MAKE) colab-$$activity || exit 1; done

colab-%: notebooks ## run one activity on a Colab T4 and collect results (make colab-a3)
	uv run python scripts/colab_run.py $*

report: ## build the PDF report
	uv run --group report python scripts/build_report.py

zip: report ## build the delivery archive
	mkdir -p entrega
	rm -f entrega/anderson_correa_computer-vision_pd.zip
	zip -j entrega/anderson_correa_computer-vision_pd.zip notebooks/A1_vision_transformers.ipynb notebooks/A2_clip_ads16.ipynb notebooks/A3_cnn_kaggle.ipynb notebooks/A4_estudo_caso_raio_x.ipynb report/anderson_correa_deep-learning-and-vision_computer-vision.pdf
	zip -r entrega/anderson_correa_computer-vision_pd.zip src/dlv tests scripts notebooks_src pyproject.toml uv.lock Makefile README.md -x '*/__pycache__/*' '*/.pytest_cache/*' '*.pyc'

clean: ## remove generated files
	rm -rf build entrega
