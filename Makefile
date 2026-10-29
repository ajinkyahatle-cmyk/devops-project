.PHONY: install lint test train train-fast mlflow run docker-build docker-run gate

install:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest -q

train:
	python -m insurance.train

train-fast:
	python -m insurance.train --fast --no-track

gate:
	python -m insurance.metrics --min-r2 0.80

mlflow:
	mlflow ui --port 5000

run:
	streamlit run app.py

docker-build: train
	docker build --build-arg GIT_SHA=local -t insurance-app .

docker-run:
	docker run --rm -p 8501:8501 insurance-app
