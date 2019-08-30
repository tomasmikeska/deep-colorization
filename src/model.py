# from keras.applications.resnet50 import ResNet50
from models.resnet34 import ResNet34
from models.unet import Unet


def build_model(img_width=256,
                img_height=256,
                pretrained_encoder=True,
                freeze_encoder=False):

    encoder = ResNet34(input_shape=(img_height, img_width, 3))

    if freeze_encoder:
        for layer in encoder.layers:
            layer.trainable = False

    return Unet(encoder)
