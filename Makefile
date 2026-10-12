.PHONY: install up up-gpu down lint test test-int pipeline pipeline-dev \
        eval-holdout eval-recall mcp-install fixture fetch-kaikki preflight clean-state \
        tensegrity tensegrity-dev clean-state-tensegrity

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

# ---------------------------------------------------------------------------
# Tensegrity pipeline — document ingestion into the `tensegrity` DB/collection.
# L15 = sentences, L14 = titles, role-tagged (tev1 complement/contradict) edges.
# Profile is always `tensegrity` (.env.tensegrity); DOC is the input file.
# ---------------------------------------------------------------------------
TSTAGES = 01_parse 02_embed 03_insert_nodes 04_l15_edges 05_title_vectors \
          06_l14_edges 07_orphan_reentry 08_metabary 09_extend 10_index

# Usage: make tensegrity DOC=path/to/file.md
tensegrity:
	@test -n "$(DOC)" || (echo "set DOC=path/to/file.md" && exit 1)
	@for s in $(TSTAGES); do \
	  echo "== $$s =="; \
	  if [ "$$s" = "01_parse" ]; then \
	    $(PY) -m tensegrity_pipeline.$$s --doc "$(DOC)" || exit 1; \
	  else \
	    $(PY) -m tensegrity_pipeline.$$s || exit 1; \
	  fi; \
	done

# Small-corpus smoke test: parse the doc, then embed/pair only the first 200
# sentences (dev only; the real run drops --limit).
tensegrity-dev:
	@test -n "$(DOC)" || (echo "set DOC=path/to/file.md" && exit 1)
	$(PY) -m tensegrity_pipeline.01_parse --doc "$(DOC)"
	$(PY) -m tensegrity_pipeline.02_embed --limit 200
	$(PY) -m tensegrity_pipeline.03_insert_nodes
	$(PY) -m tensegrity_pipeline.04_l15_edges
	$(PY) -m tensegrity_pipeline.05_title_vectors
	$(PY) -m tensegrity_pipeline.06_l14_edges
	$(PY) -m tensegrity_pipeline.07_orphan_reentry
	$(PY) -m tensegrity_pipeline.08_metabary
	$(PY) -m tensegrity_pipeline.09_extend
	$(PY) -m tensegrity_pipeline.10_index

clean-state-tensegrity:
	rm -f pipeline_state_tensegrity/*.json
	rm -f data/parsed_tensegrity/*.jsonl
	rm -f data/parsed_tensegrity/*.tmp
	rm -f data/parsed_tensegrity/missed.txt
