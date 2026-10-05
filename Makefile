.PHONY: install up up-gpu down lint test test-int pipeline pipeline-dev \
        eval-holdout eval-recall mcp-install fixture fetch-kaikki preflight clean-state

PY ?= python3

install:
	$(PY) -m pip install -e ".[dev]"

up:
	docker compose up -d mongodb

up-gpu:
	docker compose --profile gpu up -d

down:
	docker compose --profile gpu down --remove-orphans

lint:
	ruff check .
	mypy

test:
	pytest --cov=lib --cov-report=term

test-int:
	pytest -m integration -v -o addopts=''

STAGES = s01_parse s02_embed s03_insert_nodes s04_l15_edges s05_word_vectors \
         s06_l14_edges s07_orphan_reentry s08_metabary s09_extend s10_index

# Build profile: `poc` (English PoC, default) or `build-all` (multilingual).
# Maps to .env / .env.build-all; passed through BARY_ENV.
ENV ?= poc

pipeline:
	@for s in $(STAGES); do \
	  echo "== $$s =="; \
	  BARY_ENV=$(ENV) $(PY) -m scripts.$$s || exit 1; \
	done

# Small-corpus smoke test: parse only the first 15 000 kaikki lines.
# Useful for local end-to-end testing before committing to a full VPS run.
pipeline-dev:
	BARY_ENV=$(ENV) $(PY) -m scripts.s01_parse --limit 15000
	BARY_ENV=$(ENV) $(PY) -m scripts.s02_embed
	BARY_ENV=$(ENV) $(PY) -m scripts.s03_insert_nodes
	BARY_ENV=$(ENV) $(PY) -m scripts.s04_l15_edges
	BARY_ENV=$(ENV) $(PY) -m scripts.s05_word_vectors
	BARY_ENV=$(ENV) $(PY) -m scripts.s06_l14_edges
	BARY_ENV=$(ENV) $(PY) -m scripts.s07_orphan_reentry
	BARY_ENV=$(ENV) $(PY) -m scripts.s08_metabary
	BARY_ENV=$(ENV) $(PY) -m scripts.s09_extend
	BARY_ENV=$(ENV) $(PY) -m scripts.s10_index

eval-holdout:
	$(PY) -m scripts.eval.holdout

# Use --max-pairs N to cap eval length during smoke testing.
eval-recall:
	$(PY) -m scripts.eval.recall

mcp-install:
	$(PY) -m pip install -e ".[mcp]"

fixture:
	$(PY) -m scripts.dev.make_fixture

fetch-kaikki:
	bash scripts/dev/fetch_kaikki.sh

preflight:
	$(PY) -m scripts.preflight

clean-state:
	rm -f pipeline_state/*.json
	rm -f data/parsed/*.jsonl

# clean-state for the multilingual all-build
clean-state-all:
	rm -f pipeline_state_all/*.json
	rm -f data/parsed_all/*.jsonl
