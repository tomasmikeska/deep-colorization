from keras.layers import *
from keras.models import Model


def Discriminator(input_shape=(256, 256, 3), n_blocks=3):
    img_input = Input(shape=input_shape)

    x = Conv2D(256, (4, 4), padding='same', strides=2)(img_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.15 / 2)(x)

    for i in range(n_blocks):
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.15)(x)
        x = Conv2D(512, (4, 4), padding='same', strides=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(img_input, x)
