include .env
export

remote-train:
	git archive -o paperspace.zip $(shell git stash create)
	zip -u paperspace.zip .env
	gradient jobs create \
		--name "image colorization train" \
		--projectId "pryqks47v" \
		--machineType "P5000" \
		--container "tomikeska/ml-box" \
		--workspaceArchive paperspace.zip \
		--command "make train"

remote-train-gan:
	git archive -o paperspace.zip $(shell git stash create)
	zip -u paperspace.zip .env
	gradient jobs create \
		--name "image colorization train GAN" \
		--projectId "pryqks47v" \
		--machineType "P6000" \
		--container "tomikeska/ml-box" \
		--workspaceArchive paperspace.zip \
		--command "make train-gan"

train:
	pip3 install -r requirements.txt
	python3 src/train.py \
		--train-dataset=/storage/datasets/imagenet/train/ \
		--test-dataset=/storage/datasets/imagenet/val/ \
		--validation-path=/storage/datasets/colorization-val/ \
		--batch-size=32 \
		--img-w=128 \
		--img-h=128 \
		--weights=/storage/fusion_unet_01_263.405.h5 \
		--model-save-path=/artifacts/

train-gan:
	pip3 install -r requirements.txt
	python3 src/train_gan.py \
		--train-dataset=/storage/datasets/imagenet/train/ \
		--validation-path=/storage/datasets/deep-colorization/val/ \
		--batch-size=32 \
		--img-w=192 \
		--img-h=192 \
		--epoch=1 \
		--gan-weights=/storage/gan_128x128_epoch-1_15K.h5 \
		--model-save-path=/artifacts/

test:
	PYTHONPATH=src/ python -m pytest tests/

lint:
	python -m pycodestyle src/
