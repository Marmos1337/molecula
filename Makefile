PY ?= .venv/bin/python
export PYTHONPATH = src

TEST ?= data/kaggle/test.csv
SAMPLE ?= data/kaggle/sample_submit.csv

.PHONY: setup collect collect-offline rebuild audit targeted pubchem pubchem-structure experiments analysis train test serve submission info all clean

setup:                      ## create the virtualenv and install pinned dependencies
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

collect:                    ## rebuild the ChEMBL dataset (network; re-runs from cache)
	$(PY) -m molecula.collect --out data

collect-offline:            ## rebuild the ChEMBL dataset from the cached API pages only
	$(PY) -m molecula.collect --out data --offline

rebuild:                    ## rebuild the ChEMBL dataset from the raw extracts in the repo (no network)
	$(PY) -m molecula.collect --out data --from-raw

audit:                      ## validate the competition files, write reports/official_data_audit.json
	$(PY) -m molecula.official

targeted:                   ## look every competition structure up in ChEMBL
	$(PY) -m molecula.collect_targeted

pubchem:                    ## collect indices from PubChem, the organisers' own source
	$(PY) -m molecula.collect_pubchem

pubchem-structure:          ## the same by structure instead of substance id (measured worse)
	$(PY) -m molecula.collect_pubchem --by-structure

experiments:                ## compare every approach on identical folds (~1 hour)
	$(PY) -m molecula.experiments --out reports

analysis:                   ## error analysis on the out-of-fold predictions, then the report
	$(PY) -m molecula.analysis
	$(PY) scripts/render_results.py

train:                      ## refit the chosen approach on all training rows
	$(PY) -m molecula.train

test:                       ## contract, curation and competition-rule tests
	$(PY) -m unittest discover -s tests -v

serve:                      ## start the web service on http://127.0.0.1:8000
	.venv/bin/uvicorn molecula.api:app --host 127.0.0.1 --port 8000

info:                       ## print the saved model's version and measured metrics
	$(PY) -m molecula.predict info

submission:                 ## build submission.csv from the real sample file
	$(PY) -m molecula.predict submission \
		--test "$(TEST)" --sample-submission "$(SAMPLE)" \
		--target-scale si --measured-csv data/processed/pubchem.csv --no-use-measured \
		--out submission.csv

all: audit rebuild pubchem experiments analysis train submission test  ## the whole pipeline, in order

clean:
	rm -rf models/*.joblib reports/*.json submission.csv submission.report.json
