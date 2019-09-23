from keras.layers import *
from keras.models import Model
from layers.self_attention import SelfAttention
from layers.spectral_norm import ConvSN2D, DenseSN


def AttentionDiscriminator(input_shape=(256, 256, 3), n_blocks=3):
    img_input = Input(shape=input_shape)

    x = ConvSN2D(256, (4, 4), padding='same', strides=2)(img_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.15 / 2)(x)

    for i in range(n_blocks):
        x = ConvSN2D(256, (3, 3), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.15)(x)
        x = ConvSN2D(512, (4, 4), padding='same', strides=2)(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = SelfAttention(channels=512)(x)

    x = ConvSN2D(256, (3, 3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = GlobalAveragePooling2D()(x)
    x = DenseSN(1, activation='sigmoid')(x)

    return Model(img_input, x)
