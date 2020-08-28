import keras.backend as K
import tensorflow as tf
from keras import initializers, layers
import math

class Length(layers.Layer):

    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):

    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
            mask = K.expand_dims(mask, -1)
        else:
            x = K.sqrt(K.sum(K.square(inputs), -1, True))
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)

        return K.batch_flatten(inputs * mask)

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # doğru değerler sağlanır
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # doğru olmayan değerler sağlanır
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):

    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)

    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    #scale = K.log(K.exp(s_squared_norm)+0.5) / s_squared_norm
    #scale = (1-(1/K.exp(s_squared_norm))) / s_squared_norm

    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 name='digitcaps',###############
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.name = name##################

    def _build(self, input_shape): #####################build => _build
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    #this func for call model(init params)
    def get_config(self):
        # For serialization with 'custom_objects'
        config = super().get_config()
        config['num_capsule'] = self.num_capsule
        config['dim_capsule'] = self.dim_capsule
        return config
'''
    def call(self, inputs, training=None):
        inputs_expand = K.expand_dims(inputs, 1)
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)
        inputs_hat_stopped = K.stop_gradient(inputs_hat)
        b = K.stop_gradient(K.sum(K.zeros_like(inputs_hat), -1))

        assert self.num_routing > 0, 'The num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(b, dim=1)

            if i == self.num_routing - 1:

                outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))
            else:
                outputs = squash(K.batch_dot(c, inputs_hat_stopped, [2, 2]))
                b += K.batch_dot(outputs, inputs_hat_stopped, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])
'''


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):

    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)