.PHONY: install test lint dry-train dry-eval dry-play tree

install:
	pip install -e .

test:
	pytest -q

lint:
	ruff check src tests

dry-train:
	summer-camp train --task manipulation/franka_reach --config tasks/manipulation/configs/franka_reach.yaml --dry-run

dry-eval:
	summer-camp eval --task manipulation/franka_reach --config tasks/manipulation/configs/franka_reach.yaml --dry-run

dry-play:
	summer-camp play --task manipulation/franka_reach --config tasks/manipulation/configs/franka_reach.yaml --dry-run

tree:
	find . -maxdepth 4 -type f | sort
