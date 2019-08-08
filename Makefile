include .env
export

remote-train:
	git archive -o paperspace.zip $(shell git stash create)
	zip -u paperspace.zip .env
	gradient jobs create \
		--name "image colorization train" \
		--machineType "P5000" \
		--container "tomikeska/ml-box" \
		--workspaceArchive paperspace.zip \
		--ports $(TB_PORT):$(TB_PORT) \
		--command "make train"

train:
	pip3 install -r requirements.txt
	python3 src/train.py

local-train:
	python src/train.py

test:
	PYTHONPATH=src/ python -m pytest tests/
