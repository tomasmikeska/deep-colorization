import keras.backend as K
from keras.layers import *
from keras.models import Model
from layers.self_attention import SelfAttention
from layers.spectral_norm import ConvSN2D


def conv_block(x, filters=64, batch_norm=False, downsample=False):
    x = ConvSN2D(filters, (3, 3), padding='same', strides=2 if downsample else 1)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def up_block(x, y, filters=64, batch_norm=False):
    x = UpSampling2D((2, 2))(x)
    x = ConvSN2D(filters, (3, 3), padding='same')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([x, y])
    x = ConvSN2D(filters, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = ConvSN2D(filters, (3, 3), padding='same')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def AttentionUNet(input_shape=(64, 64, 3)):
    img_input = Input(shape=input_shape)

    # Encoder
    enc1 = conv_block(img_input, 64)
    down = conv_block(enc1, 64, downsample=True)

    enc2 = conv_block(down, 128)
    down = conv_block(enc2, 128, downsample=True)

    enc3 = conv_block(down, 256)
    down = conv_block(enc3, 256, downsample=True)

    enc4 = conv_block(down, 512)
    down = conv_block(enc4, 512, downsample=True)

    enc5 = conv_block(down, 1024)
    enc5 = conv_block(enc5, 1024)

    fus = Conv2D(256, (1, 1))(enc5)
    fus = ReLU()(fus)

    # Decoder
    dec = up_block(fus, enc4, 512)
    dec = SelfAttention(channels=512)(dec)

    dec = up_block(dec, enc3, 256)
    dec = SelfAttention(channels=256)(dec)

    dec = up_block(dec, enc2, 128)
    dec = SelfAttention(channels=128)(dec)

    dec = up_block(dec, enc1, 64)

    out = Conv2D(3, (1, 1), activation='tanh')(dec)

    return Model(img_input, out)
