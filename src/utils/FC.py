import os
from keras.layers.core import Layer
from keras.layers import concatenate
import keras.backend as K
from keras.models import Model
from keras.layers import Input

class FC(Layer):
    def __init__(self, name, **kwargs):
        self.name = name
        super(FC, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.kernel = []
        for i in range(input_shape[1]):
            self.kernel.append(self.add_weight(shape=(2, 1), initializer='glorot_uniform', name=self.name + "_kernel_" + str(i+1)))
        
    def call(self, inputs, mask=None):
        _, classes, _ = K.int_shape(inputs)
        output = []
        for i in range(classes):
            output.append(K.dot(inputs[:, i], self.kernel[i]))
        return concatenate(output, axis=-1)
    
    def compute_output_shape(self, input_shape):
        #print(input_shape)
        return (input_shape[0], input_shape[1])

    def get_config(self):
        config = {}#
        base_config = super(FC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    inpt = Input(shape=(17, 2))
    oupt = FC("0")(inpt)
    model = Model(inpt, oupt, name='Inception')