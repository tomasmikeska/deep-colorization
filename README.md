# Deep colorization

Implementation of deep learning model in Keras for image colorization. Project uses U-Net trained as Self-Attention GAN together with Perceptual loss instead of usual MAE or MSE. Work is still in progress.

---

### Latest Examples

Historical photos

![](https://i.imgur.com/64DjM7E.jpg)
*"The Roaring Lion", Winston Churchill's iconic portrait, 1941*

![](https://i.imgur.com/bTGzCDg.jpg)
*Lower Manhattan’s Classic Skyline Seen Aerially From Battery Park, 1956*

![](https://i.imgur.com/trLCobT.jpg)
*"Migrant Mother" by Dorothea Lange, 1936*

![](https://i.imgur.com/G6tSaaW.jpg)
*NYC street vintage photo*

![](https://i.imgur.com/EzJFUJq.jpg)
*Metropolis (movie), 1927*

Original RGB images (left) converted to grayscale and colorized (right)

![](https://i.imgur.com/CYgH4K3.jpg)

![](https://i.imgur.com/hV4wsDE.jpg)

![](https://i.imgur.com/Yskc3iL.jpg)

---

### Requirements

- Python 3.x
- pip

### Installation and setup

Install pip packages using
```
$ pip install -r requirements.txt
```

Add `.env` file to project root with environmental variables
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

### Usage

Train model using command
```
$ python src/train_gan.py
```

Colorize image using trained weights
```
$ python src/colorize.py --weights model/weights.h5 --source source.jpg --output output.jpg
```

### License

Code is released under the MIT License. Please see the LICENSE file for details.
