from keras.layers import *
from keras.models import Model


def res_block(inputs, filters=64):
    x = Conv2D(filters, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, inputs])
    x = ReLU()(x)
    return x


def ResNet34(input_shape):
    img_input = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=2, padding='same')(img_input)
    x = MaxPooling2D()(x)

    x = res_block(x, filters=64)
    x = res_block(x, filters=64)
    x = res_block(x, filters=64)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = res_block(x, filters=128)
    x = res_block(x, filters=128)
    x = res_block(x, filters=128)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = res_block(x, filters=256)
    x = res_block(x, filters=256)
    x = res_block(x, filters=256)
    x = res_block(x, filters=256)
    x = res_block(x, filters=256)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = res_block(x, filters=512)
    x = res_block(x, filters=512)

    return Model(img_input, x)
