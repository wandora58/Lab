
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Lambda

import keras
import numpy as np

class Cx2DAffine(Layer):

  def __init__(self, output_dim, activation, **kwargs):
    self.output_dim = output_dim
    self.activation = activation
    super(Cx2DAffine, self).__init__(**kwargs)

  def build(self, input_shape):

    # input shape: [batchNum][Real/Imag][Data]
    self.weight_real = self.add_weight(name='weight_real',
                                  shape=(input_shape[2], self.output_dim),
                                  initializer='glorot_uniform')

    self.weight_imag = self.add_weight(name='weight_imag',
                                  shape=(input_shape[2], self.output_dim),
                                  initializer='glorot_uniform')

    self.bias_real = self.add_weight(name='bias_real',
                                  shape=(1, self.output_dim),
                                  initializer='zeros')

    self.bias_imag = self.add_weight(name='bias_imag',
                                  shape=(1, self.output_dim),
                                  initializer='zeros')

    super(Cx2DAffine, self).build(input_shape)


  def call(self, x):

    # input shape: [batchNum][Real/Imag][Data]
    x_real = Lambda(lambda x: x[:, 0, :], output_shape=(x.shape[2], ))(x) # real
    x_imag = Lambda(lambda x: x[:, 1, :], output_shape=(x.shape[2], ))(x) # imag

    if self.activation == 'relu':

        # Calc: [batchNum][Output]
        real = K.dot(x_real, self.weight_real) - K.dot(x_imag, self.weight_imag)
        imag = K.dot(x_real, self.weight_imag) + K.dot(x_imag, self.weight_real)

        real = K.relu(real + self.bias_real)
        imag = K.relu(imag + self.bias_imag)

        # Expand: [batchNum][1][Output]
        real = K.expand_dims(real, 1)
        imag = K.expand_dims(imag, 1)

        # Merge: [batchNum][2][Output]
        cmpx = keras.layers.concatenate([real, imag], axis=1)
        return cmpx

    else:

        # Calc: [batchNum][Output]
        real = K.dot(x_real, self.weight_real) - K.dot(x_imag, self.weight_imag)
        imag = K.dot(x_real, self.weight_imag) + K.dot(x_imag, self.weight_real)

        real = real + self.bias_real
        imag = imag + self.bias_imag

        # Expand: [batchNum][1][Output]
        real = K.expand_dims(real, 1)
        imag = K.expand_dims(imag, 1)

        # Merge: [batchNum][2][Output]
        cmpx = keras.layers.concatenate([real, imag], axis=1)
        return cmpx


  def compute_output_shape(self, input_shape):

    #  (input_shape[0], self.output_dim)
    # Unpack tuple (BatchNum, (2, DataNum)) -> (BatchNum, 2, DataNum)
    return(input_shape[0], 2, self.output_dim)


def conv(input, chs):
    x = input
    for i in range(1):
        x = Conv1D(chs, 2, strides=1)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
    return x

