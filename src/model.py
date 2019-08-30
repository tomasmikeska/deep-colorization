from models.resnet34 import ResNet34
from models.unet import Unet


def build_model(img_width=256,
                img_height=256):
    '''
    Build colorization model - U-Net with ResNet34 down-sampling part

    Args:
        img_width (int): Input image width
        img_height (int): Input image height
    '''
    encoder = ResNet34(input_shape=(img_height, img_width, 3))
    return Unet(encoder)
