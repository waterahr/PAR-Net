import os
import sys
sys.path.append("../")
from keras.layers.core import Layer
from keras import backend as K
import random
import tensorflow as tf

from utils.mask_operation import tf_mask


class Masking(Layer):
    def __init__(self, labels, **kwargs):
        self.labels = labels
        self.shape = None
        self.supports_masking = True
        super(Masking, self).__init__(**kwargs)
        
    def compute_mask(self, inputs, input_mask=None):
        # need not to pass the mask to next layers
        return None
        
    def build(self, input_shape):
        # print("****", input_shape)### (None, None, None, 2048)
        # self.shape = input_shape
        batch_size = input_shape[0]
        h_axis, w_axis = 1, 2
        height, width = input_shape[h_axis], input_shape[w_axis]
        channel = input_shape[-1]
        self.shape = (batch_size, height, width, channel)
        
        # self.trainable_weights = [self.kernelWeights]
        
    def call(self, inputs, mask=None):
        # print(self.shape)### (None, None, None, 2048)
        return tf_mask(inputs, self.labels, 1, self.mag)
    
    def compute_output_shape(self, input_shape):
        # print("****", input_shape)
        batch_size = input_shape[0]
        h_axis, w_axis = 1, 2
        height, width = input_shape[h_axis], input_shape[w_axis]
        channel = input_shape[-1]
        self.shape = (batch_size, height, width, channel)
        return input_shape

    def get_config(self):
        config = {}###{"lambda" : self.lamda, "mag" : self.mag}
        base_config = super(Masking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))