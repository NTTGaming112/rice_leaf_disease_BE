import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.utils import register_keras_serializable

# Squash activation

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1. + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

@register_keras_serializable()
class CAR_CFL_Layer(layers.Layer):
    def __init__(self, num_capsules, capsule_dim, **kwargs):
        super(CAR_CFL_Layer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
    def build(self, input_shape):
        self.num_caps_in = input_shape[1]
        self.dim_caps_in = input_shape[2]
        self.W_ij = self.add_weight(
            shape=(1, self.num_caps_in, self.num_capsules, self.capsule_dim, self.dim_caps_in),
            initializer='glorot_uniform', name='W_ij', trainable=True
        )
        self.W_cf = self.add_weight(
            shape=(1, self.num_caps_in, self.num_capsules, self.capsule_dim, 1 + self.capsule_dim),
            initializer='glorot_uniform', name='W_cf', trainable=True
        )
        self.built = True
    def call(self, inputs):
        a_i_expanded = K.expand_dims(K.expand_dims(inputs, 2), 3)
        p_j = K.sum(a_i_expanded * self.W_ij, axis=-1)
        b_ij = tf.norm(p_j, axis=-1)
        C_ij = tf.nn.softmax(b_ij, axis=2)
        C_ij_expanded = K.expand_dims(C_ij, -1)
        p_j_concat = tf.concat([C_ij_expanded, p_j], axis=-1)
        p_j_concat_expanded = K.expand_dims(p_j_concat, -2)
        F_j = K.sum(p_j_concat_expanded * self.W_cf, axis=-1)
        a_i_broadcast = K.expand_dims(inputs, 2)
        a_i_tiled = tf.tile(a_i_broadcast, [1, 1, self.num_capsules, 1])
        a_i_oplus_F_j = tf.concat([a_i_tiled, F_j], axis=-1)
        C_ij_weighted = K.expand_dims(C_ij, -1)
        weighted_vector = C_ij_weighted * a_i_oplus_F_j
        s_j = K.sum(weighted_vector, axis=1)
        S = squash(s_j)
        return S
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_capsules, self.dim_caps_in + self.capsule_dim)
