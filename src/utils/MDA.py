from keras.layers.core import Layer
from keras.layers import Conv2D, concatenate
import keras.backend as K

class MDA(Layer):
    def __init__(self, attention_channel, feature_channel, **kwargs):
        #self.attention_channel = attention_channel
        self.attention_channel = attention_channel
        self.feature_channel = feature_channel
        super(MDA, self).__init__(**kwargs)
        
    #def build(self, input_shape):  
        
    def call(self, inputs, mask=None):
        attention = inputs[0]
        feature_map = inputs[1]
        refined_features = []
        for i in range(self.attention_channel):
            #print(K.tile(attention[..., i:i+1], [1,1,1,self.feature_channel]))
            y = K.tile(attention[..., i:i+1], [1,1,1,self.feature_channel]) * feature_map
            refined_features.append(y)
        refined_features = concatenate(refined_features, axis=-1)
        #print(refined_features)
        return refined_features
    
    def compute_output_shape(self, input_shape):
        #print(input_shape)
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.feature_channel * self.attention_channel)

    def get_config(self):
        config = {"attention_channel":self.attention_channel,
               "feature_channel":self.feature_channel}#
        base_config = super(MDA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))