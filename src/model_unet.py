from keras.layers import *
from keras.models import Model
from layers.pixel_shuffle import PixelShuffle


def build_model(img_width=256, img_height=256):
    inputs = Input((img_height, img_width, 1))

    # Down-sampling part
    conv1 = Conv2D(32, 3, padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(0.2)(conv1)

    conv1 = Conv2D(32, 3, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(0.2)(conv1)

    conv2 = Conv2D(64, 3, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(0.2)(conv2)

    conv2 = Conv2D(64, 3, strides=(2,2), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(0.2)(conv2)

    conv3 = Conv2D(128, 3, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(0.2)(conv3)

    conv3 = Conv2D(128, 3, strides=(2,2), padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(0.2)(conv3)

    conv4 = Conv2D(256, 3, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(0.2)(conv4)

    conv4 = Conv2D(256, 3, strides=(2,2), padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(0.2)(conv4)

    # Up-sampling part
    up7 = PixelShuffle()(conv4)
    up7 = concatenate([up7, conv3], axis=-1)
    conv7 = Conv2D(128, 3, padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Dropout(0.2)(conv7)

    conv7 = Conv2D(128, 3, padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Dropout(0.2)(conv7)

    up8 = PixelShuffle()(conv7)
    up8 = concatenate([up8, conv2], axis=-1)
    conv8 = Conv2D(64, 3, padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(64, 3, padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = PixelShuffle()(conv8)
    up9 = concatenate([up9, conv1], axis=-1)
    conv9 = Conv2D(32, 3, padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv9 = Conv2D(32, 3, padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(2, (1, 1), activation='tanh')(conv9)

    model = Model(inputs, conv10)

    return model
