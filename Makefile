include .env
export

remote-train:
	git archive -o paperspace.zip $(shell git stash create)
	zip -u paperspace.zip .env
	gradient jobs create \
		--name "image colorization train" \
		--machineType "P4000" \
		--container "tomikeska/ml-box" \
		--workspaceArchive paperspace.zip \
		--command "make train"

train:
	pip3 install -r requirements.txt
	python3 src/train.py \
		--train-dataset=/storage/datasets/imagenet/train/ \
		--test-dataset=/storage/datasets/imagenet/val/ \
		--batch-size=128 \
		--img-w=64 \
		--img-h=64 \
		--model-save-path=/artifacts/

local-train:
	python src/train.py

test:
	PYTHONPATH=src/ python -m pytest tests/
