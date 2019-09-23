from keras.layers import *
from keras.models import Model
from keras.applications.xception import Xception
import keras.backend as K


class FusionLayer(Layer):

    def call(self, inputs):
        imgs, embs = inputs
        reshaped_shape = imgs.shape[1:3].concatenate(embs.shape[1])
        embs = K.repeat(embs, imgs.shape[1] * imgs.shape[2])
        embs = Reshape(reshaped_shape)(embs)
        return K.concatenate([imgs, embs])

    def compute_output_shape(self, input_shapes):
        # Must have 2 tensors as input
        assert input_shapes and len(input_shapes) == 2
        imgs_shape, embs_shape = input_shapes
        # The batch size of the two tensors must match
        assert imgs_shape[0] == embs_shape[0]
        # (batch_size, width, height, embedding_len + depth)
        return imgs_shape[:3] + (imgs_shape[3] + embs_shape[1],)


def conv_block(x, filters=64, batch_norm=False):
    x = Conv2D(64, (3, 3), padding='same')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def up_block(x, y, filters=64, batch_norm=False):
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Concatenate()([x, y])
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def FusionUNet(input_shape=(64, 64, 3)):
    img_input = Input(shape=input_shape)

    # Encoder
    enc1 = conv_block(img_input, 64)
    enc1 = conv_block(enc1, 64)
    down = MaxPooling2D()(enc1)

    enc2 = conv_block(down, 128)
    enc2 = conv_block(enc2, 128)
    down = MaxPooling2D()(enc2)

    enc3 = conv_block(down, 256)
    enc3 = conv_block(enc3, 256)
    down = MaxPooling2D()(enc3)

    enc4 = conv_block(down, 512)
    enc4 = conv_block(enc4, 512)
    down = MaxPooling2D()(enc4)

    enc5 = conv_block(down, 1024)
    enc5 = conv_block(enc5, 1024)

    # Fusion
    xception = Xception(input_tensor=img_input,
                        include_top=True,
                        weights='imagenet')
    for l in xception.layers:
        l.trainable = False

    fus = FusionLayer()([enc5, xception.output])
    fus = Conv2D(256, (1, 1))(fus)
    fus = ReLU()(fus)

    # Decoder
    dec = up_block(fus, enc4, 512)
    dec = up_block(dec, enc3, 256)
    dec = up_block(dec, enc2, 128)
    dec = up_block(dec, enc1, 64)

    out = Conv2D(3, (1, 1), activation='tanh')(dec)

    return Model(img_input, out)
