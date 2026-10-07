.PHONY: install lint format test test-all study study-quick train serve ui docker clean

install:        ## editable install with every extra
	pip install -e ".[all]"

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

test:           ## fast unit tests
	pytest -m "not slow"

test-all:
	pytest --cov=heartrisk --cov-report=term-missing

study:          ## full study -> reports/REPORT.md (about 20 min on 2 cores)
	heartrisk study

study-quick:
	heartrisk study --quick --out /tmp/heartrisk-reports --results /tmp/heartrisk-results

train:
	heartrisk train

serve: train
	heartrisk serve

ui: train
	streamlit run app/streamlit_app.py

docker:
	docker compose up --build

clean:
	rm -rf results .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
