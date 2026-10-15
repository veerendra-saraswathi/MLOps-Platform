.PHONY: help up down train api register

help:
	@echo "make up       - start MLflow + MinIO"
	@echo "make down     - stop containers"
	@echo "make train    - run the iris training example"
	@echo "make api      - start the FastAPI control plane"
	@echo "make register - example: make register NAME=iris RUN_ID=xxx"

up:
	docker compose up -d

down:
	docker compose down

train:
	python examples/train_and_serve.py

api:
	uvicorn src.api.main:app --reload --port 8000

register:
	python examples/register_model.py --name $(NAME) --run-id $(RUN_ID)