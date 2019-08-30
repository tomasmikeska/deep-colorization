# WIP Deep colorization

Implementation of deep learning model in Keras for image colorization. Project uses U-Net with ResNet34 encoder,
PixelShuffle upsample layers with ICNR initialization and feature loss from [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) instead of usual MAE or MSE.

#### Requirements

- Python 3.x
- pip

#### Installation and setup

Install pip packages using
```
$ pip install -r requirements.txt
```

Add .env file to project root with environmental variables
```
COMET_PROJECTNAME={comet_project_name}
COMET_WORKSPACE={comet_workspace}
COMET_API_KEY={comet_api_key}
```

[optional]

There is a Docker image included that was used for training in cloud. You can build it from local Dockerfile with
```
docker build -t ml-box .
```
or get it from Docker Hub
```
docker pull tomikeska/ml-box
```

#### Usage

Train model using command
```
$ python src/train.py
```

Colorize image using trained weights
```
$ python src/colorize.py --model model/weights.h5 --source source.jpg --output output.jpg
```
