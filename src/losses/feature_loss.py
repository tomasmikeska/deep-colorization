import keras.backend as K
from keras.models import Model
from keras.layers import MaxPooling2D
from keras.applications.vgg16 import VGG16


def l1_distance(a, b):
    return K.mean(K.abs(a - b))


def create_feature_loss(input_shape=(64, 64, 3),
                        loss_weights=[20, 70, 10]):
    # Base feature loss model
    vgg16 = VGG16(input_shape=input_shape,
                  include_top=False,
                  weights='imagenet')
    # Convert to model with output of 3 hook layers
    blocks = [i - 1 for i, l in enumerate(vgg16.layers) if isinstance(l, MaxPooling2D)]  # All blocks before maxpooling
    layer_ids = blocks[2:5]  # Filter only the higher layers for colorization
    loss_features = [vgg16.layers[i] for i in layer_ids]
    loss_model = Model(vgg16.inputs, list(map(lambda l: l.output, loss_features)))

    def feature_loss(y_true, y_pred):
        in_feat = loss_model(y_true)
        out_feat = loss_model(y_pred)
        loss = l1_distance(y_true, y_pred)
        loss += [l1_distance(f_in, f_out) * w for f_in, f_out, w in zip(in_feat, out_feat, loss_weights)]
        return K.sum(loss)

    return feature_loss
