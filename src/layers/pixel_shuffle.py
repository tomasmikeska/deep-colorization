import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import Layer


class PixelShuffle(Layer):
    def __init__(self, r=2, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.r = r

    def call(self, I):
        r = self.r
        _, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0]
        X = K.reshape(I, [bsize, a, b, int(c/(r*r)),r, r])  # (bsize, a, b, c/(r*r), r, r)
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # (bsize, a, b, r, r, c/(r*r))
        X = [X[:,i,:,:,:,:] for i in range(a)]  # [ (bsize, b, r, r, c/(r*r)), ... ]
        X = K.concatenate(X, 2)  # (bsize, b, a*r, r, c/(r*r))
        X = [X[:,i,:,:,:] for i in range(b)] # [ (bsize, a*r, r, c/(r*r)), ... ]
        X = K.concatenate(X, 2)  # (bsize, a*r, b*r, c/(r*r))
        return X

    def compute_output_shape(self, input_shape):
        r = self.r
        batch_size, a, b, c = input_shape
        return (batch_size, a*r, b*r, c // (r*r))

    def get_config(self):
        config = super(Layer, self).get_config()
        config['r'] = self.r
        return config
