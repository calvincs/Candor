PY ?= .venv/bin/python
PYTEST = $(PY) -m pytest

.PHONY: help venv unit conformance stage1 stage2 stage3 stage4 stage5 gates audit clean examples claims claims-fast all-tests

help:
	@echo "CANDOR — spec v0.2 conformance targets"
	@echo "  make venv         create .venv and install pytest + hypothesis"
	@echo "  make stage1..5    run one §6.7 stage gate"
	@echo "  make gates        run stage1..stage4 in build order, stopping at the first red"
	@echo "  make unit         additive unit suite (tests/unit)"
	@echo "  make claims       validate the public claims on synthetic worlds"
	@echo "  make conformance  the whole harness"
	@echo "  make audit        the two source-tree invariants (§6.2)"

venv:
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q pytest hypothesis

stage1:
	$(PYTEST) tests/conformance.py -m stage1

stage2:
	$(PYTEST) tests/conformance.py -m stage2

stage3:
	$(PYTEST) tests/conformance.py -m stage3

stage4:
	$(PYTEST) tests/conformance.py -m stage4

stage5:
	$(PYTEST) tests/conformance.py -m stage5

# §8: never start stage N+1 while stage N is red.
gates: stage1 stage2 stage3 stage4

unit:
	$(PYTEST) tests/unit

# §6.9 claims suite: the public promises, measured on planted synthetic worlds.
# CLAIMS_SCALE raises the replication count (1 = CI-sized, 4-8 = investigation).
claims:
	$(PYTEST) tests/claims -q

claims-fast:
	$(PYTEST) tests/claims -q -m "not slow"

all-tests: gates stage5 unit claims

conformance:
	$(PYTEST) tests/conformance.py

audit:
	$(PY) -c "import sys; sys.path.insert(0,'src'); from candor import audit; \
	  print('retrieval -> counts import paths:', audit.retrieval_writer_import_paths()); \
	  print('weight outside committed tier:', audit.grep_weight_outside_committed())"

clean:
	rm -rf .pytest_cache .hypothesis
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

examples:
	$(PY) examples/quickstart.py
	$(PY) examples/source_reliability.py
	$(PY) examples/regime_change.py
	$(PY) examples/categorical.py
	$(PY) examples/axiom_loops.py
