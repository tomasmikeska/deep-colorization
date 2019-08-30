import tensorflow as tf
import keras.backend as K
from keras.layers import *
from keras.models import Model
from layers.pixel_shuffle import PixelShuffle


def get_hook_indexes(model):
    hook_idxs = []
    for i in range(1, len(model.layers)):
        if model.layers[i - 1].output_shape[1] != model.layers[i].output_shape[1]:
            if isinstance(model.layers[i - 1], ZeroPadding2D):
                continue
            hook_idxs.append(i - 1)
    return reversed(hook_idxs)


def conv_block(x, filters=64):
    x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def icnr_initializer(initializer=tf.glorot_uniform_initializer(), scale=2):
    def init(shape, dtype=None):
        new_shape = shape[:3] + (shape[3] // (scale ** 2),)
        x = initializer(new_shape, dtype)
        x = tf.transpose(x, perm=[2, 0, 1, 3])
        x = tf.image.resize_nearest_neighbor(x, size=(shape[0] * scale, shape[1] * scale))
        x = tf.space_to_depth(x, block_size=scale)
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        return x
    return init


def pixel_shuffle_icnr(x, filters=64):
    x = Conv2D(filters, 3,
               padding='same',
               kernel_initializer=icnr_initializer(),
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = PixelShuffle()(x)
    return x


def unet_block(x, y, filters=64):
    x = pixel_shuffle_icnr(x, filters=filters)
    x = conv_block(x, filters=K.int_shape(y)[-1])
    x = Concatenate()([x, y])
    x = ReLU()(x)
    x = conv_block(x, filters=filters)
    return x


def Unet(encoder, fusion_dim=None):
    hook_idxs = get_hook_indexes(encoder)
    enc = encoder.output

    if fusion_dim is not None:
        enc = Conv2D(fusion_dim, 1, name='fusion_conv')(enc)
        enc = BatchNormalization()(enc)
        enc = ReLU()(enc)
    x = conv_block(enc, filters=K.int_shape(enc)[-1])

    for i, hook_idx in enumerate(hook_idxs):
        y = encoder.layers[hook_idx].output
        x = unet_block(x, y, filters=max(64, 512 // 2**i))

    x = conv_block(x, filters=128)
    x = Conv2D(3, 1, activation='tanh', use_bias=False)(x)

    return Model(encoder.input, x)
