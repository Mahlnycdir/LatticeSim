PY ?= .venv/bin/python

.PHONY: test test-all studies figures docs clean

test:             ## fast tests
	$(PY) -m pytest

test-all:         ## all tests, including the slow ones
	$(PY) -m pytest -m "slow or not slow"

studies:          ## run all studies, results -> results/
	$(PY) -m studies

figures:          ## render figures that are out of date -> figures/
	$(PY) -m studies.figures

docs:             ## documentation -> docs/build/latticesim.pdf (runs missing studies first)
	$(PY) tools/build_docs.py --studies

clean:            ## remove generated files (results, figures, documentation build)
	rm -rf results figures docs/build docs/generated
