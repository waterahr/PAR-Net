import os
import numpy as np
import sys
sys.path.append("../")
import keras
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D, concatenate, concatenate, Activation, Lambda, Dropout, BatchNormalization, Embedding, Reshape
#from utils.mask_operation import tf_mask

from utils.mask_layer import Masking
from utils.MDA import MDA
from utils.FC import FC
from utils.channel_pool import max_out, ave_out, sum_out


class GoogLeNet:
    
    predictions_low = []
    predictions_mid = []
    predictions_hig = []
    
    @staticmethod
    def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1,1), name=None, trainable=True):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name
        else:
            bn_name = None
            conv_name = None
     
        x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name, trainable=trainable)(x)
        x = BatchNormalization(axis=3, name=bn_name, trainable=trainable)(x)
        return x
 
    @staticmethod
    def Inception(x, nb_filter, name=None, trainable=True):
        """
        branch1x1 = GoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
     
        branch3x3 = GoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
        branch3x3 = GoogLeNet.Conv2d_BN(branch3x3, nb_filter,(3,3), padding='same', strides=(1,1), name=name)
     
        branch5x5 = GoogLeNet.Conv2d_BN(x, nb_filter, (1,1), padding='same', strides=(1,1),name=name)
        branch5x5 = GoogLeNet.Conv2d_BN(branch5x5, nb_filter, (5,5), padding='same', strides=(1,1), name=name)
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
        branchpool = GoogLeNet.Conv2d_BN(branchpool, nb_filter, (1,1), padding='same', strides=(1,1), name=name)
        """
        branch1x1 = GoogLeNet.Conv2d_BN(x, nb_filter[0], (1,1), padding='same', strides=(1,1), name=name+'_1x1', trainable=trainable)
     
        branch3x3 = GoogLeNet.Conv2d_BN(x, nb_filter[1], (1,1), padding='same', strides=(1,1), name=name+'_3x3_reduce', trainable=trainable)
        branch3x3 = GoogLeNet.Conv2d_BN(branch3x3, nb_filter[2],(3,3), padding='same', strides=(1,1), name=name+'_3x3', trainable=trainable)
     
        branch5x5 = GoogLeNet.Conv2d_BN(x, nb_filter[3], (1,1), padding='same', strides=(1,1),name=name+'5x5_reduce', trainable=trainable)
        branch5x5 = GoogLeNet.Conv2d_BN(branch5x5, nb_filter[4], (5,5), padding='same', strides=(1,1), name=name+'_5x5', trainable=trainable)
     
        branchpool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', trainable=trainable)(x)
        branchpool = GoogLeNet.Conv2d_BN(branchpool, nb_filter[5], (1,1), padding='same', strides=(1,1), name=name+'_pool_proj', trainable=trainable)
     
        x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis=3)
     
        return x
    
    @staticmethod
    def part_attention(x, classes, version="v1", tri_loss=False, center_loss=False, name=""):
        if version == "v1":
            ###########################################Part Max&Ave Pool Softmax###########################################
            #"""    
            _, h_dim, w_dim, c_dim = K.int_shape(x)
            max_pool = Lambda(lambda x:max_out(x, 1))(x)
            ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
            con_pool = concatenate([max_pool, ave_pool], axis=-1)
            w = Conv2D(len(classes), (1, 1))(con_pool)
            w = Lambda(lambda x:keras.activations.softmax(x, axis=-1))(w)
            preds = []
            for i in range(len(classes)):
                w_ = Lambda(lambda x:K.tile(x[..., i:i+1], [1,1,1,c_dim]))(w)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w_])
                # print(refined_x)
                refined_x = GlobalAveragePooling2D()(refined_x)
                # print(refined_x)
                y = Dense(classes[i], activation='sigmoid', name=name+"_fc_part_"+str(i+1))(refined_x)
                # print(y)
                preds.append(y)
            
            if center_loss:
                input_target = Input(shape=(len(classes),))
            #max_pool = Lambda(lambda x:max_out(x, 1))(x)
            #ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
            #con_pool = concatenate([max_pool, ave_pool], axis=-1)
            #c_dim = K.int_shape(con_pool)[-1]
            ws = []
            for i in range(len(classes)):
                w = GlobalAveragePooling2D()(x)
                w = Dense(c_dim // 16, activation="relu")(w)
                w = Dense(c_dim, activation="sigmoid")(w)
                #print(w)
                #print(w)
                ws.append(w)
                
                if center_loss:
                    centers = Embedding(len(classes), c_dim)(Lambda(lambda x:K.expand_dims(x[:, i], axis=1))(input_target)) #Embedding层用来存放中心
                    #print(centers)
                    if i == 0:
                        l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name=name+'_l2_loss'+str(i))([w, centers])
                    else:
                        new_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name=name+'_new_loss'+str(i))([w, centers])
                        l2_loss = Lambda(lambda x:x[0] + x[1], name = name+'_l2_loss' + str(i))([l2_loss, new_loss])
                        #print(l2_loss)
                        
                w = Lambda(lambda x:K.tile(K.expand_dims(K.expand_dims(x, axis=1), axis=1), [1,h_dim,w_dim,1]))(w)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w])
                refined_x = GlobalAveragePooling2D()(refined_x)
                refined_x = Dropout(0.4)(refined_x)
                refined_x = Dense(1024, activation='relu')(refined_x)
                y = Dense(classes[i], activation='sigmoid', name=name+"_fc2_part_"+str(i+1))(refined_x)
                if name == "low":
                    GoogLeNet.predictions_low.append(Lambda(lambda x:(x[0] + x[1]) / 2)([preds[i], y]))
                elif name == "mid":
                    GoogLeNet.predictions_mid.append(Lambda(lambda x:(x[0] + x[1]) / 2)([preds[i], y]))
                elif name == "hig":
                    GoogLeNet.predictions_hig.append(Lambda(lambda x:(x[0] + x[1]) / 2)([preds[i], y]))
                """
                yy = []
                for j in range(classes[i]):
                    yy.append(Dense(1, activation='linear', name="dense_part_"+str(i+1)+"_"+str(j+1))(concatenate([preds[i, :, j:j+1], y[:, j:j+1]], axis=-1)))
                predictions.append(concatenate(yy, axis=-1))
                #"""
                
    @staticmethod
    def part_attention_spatial(x, classes, version, name):
        _, h_dim, w_dim, c_dim = K.int_shape(x)
        if version == "v2":
            max_pool = Lambda(lambda x:max_out(x, 1))(x)
            ave_pool = Lambda(lambda x:ave_out(x, 1))(x)
            con_pool = concatenate([max_pool, ave_pool], axis=-1)
            w = Conv2D(len(classes), (1, 1))(con_pool)
            w = Lambda(lambda x:keras.activations.softmax(x, axis=-1))(w)
            
            for i in range(len(classes)):
                w_ = Lambda(lambda x:K.tile(x[..., i:i+1], [1,1,1,c_dim]))(w)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w_])
                # print(refined_x)
                refined_x = GlobalAveragePooling2D()(refined_x)
                # print(refined_x)
                y = Dense(classes[i], activation='sigmoid', name=name+"_fc_part_"+str(i+1))(refined_x)
                # print(y)
                if name == "low":
                    GoogLeNet.predictions_low.append(y)
                elif name == "mid":
                    GoogLeNet.predictions_mid.append(y)
                elif name == "hig":
                    GoogLeNet.predictions_hig.append(y)
    
    @staticmethod
    def part_attention_channel(x, classes, version, name, tri_loss=False, center_loss=False):
        _, h_dim, w_dim, c_dim = K.int_shape(x)
        if version == "v2":
            if center_loss:
                input_target = Input(shape=(len(classes),))
            ws = []
            for i in range(len(classes)):
                w = GlobalAveragePooling2D()(x)
                w = Dense(c_dim // 16, activation="relu")(w)
                w = Dense(c_dim, activation="sigmoid")(w)
                ws.append(w)
                
                if center_loss:
                    centers = Embedding(len(classes), c_dim)(Lambda(lambda x:K.expand_dims(x[:, i], axis=1))(input_target)) #Embedding层用来存放中心
                    if i == 0:
                        l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name='l2_loss'+str(i))([w, centers])
                    else:
                        new_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:, 0]), 1, keepdims=True), name='new_loss'+str(i))([w, centers])
                        l2_loss = Lambda(lambda x:x[0] + x[1], name = 'l2_loss' + str(i))([l2_loss, new_loss])
                        #print(l2_loss)
                        
                w = Lambda(lambda x:K.tile(K.expand_dims(K.expand_dims(x, axis=1), axis=1), [1,h_dim,w_dim,1]))(w)
                refined_x = Lambda(lambda x:x[0] * x[1])([x, w])
                refined_x = GlobalAveragePooling2D()(refined_x)
                refined_x = Dropout(0.4)(refined_x)
                refined_x = Dense(1024, activation='relu')(refined_x)
                y = Dense(classes[i], activation='sigmoid', name=name+"_fc_part_"+str(i+1))(refined_x)
                if name == "low":
                    GoogLeNet.predictions_low.append(y)
                elif name == "mid":
                    GoogLeNet.predictions_mid.append(y)
                elif name == "hig":
                    GoogLeNet.predictions_hig.append(y)

    @staticmethod
    def orig_build(width, height, depth, classes, weights="imagenet"):
        inpt = Input(shape=(width, height, depth))
        #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = GoogLeNet.Conv2d_BN(inpt, 64, (7,7), strides=(2,2), padding='same', name="conv1_7x7_s2_g", trainable=True)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        x = GoogLeNet.Conv2d_BN(x, 192, (3,3), strides=(1,1), padding='same', name="conv2_3x3_g", trainable=True)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        """
        x = GoogLeNet.Inception(x, 64, name="inception_3a")#256
        x = GoogLeNet.Inception(x, 120, name="inception_3b")#480
        """
        x = GoogLeNet.Inception(x, [64,96,128,16,32,32], name="inception_3a_g", trainable=True)#256
        x = GoogLeNet.Inception(x, [128,128,192,32,96,64], name="inception_3b_g", trainable=True)#480
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        """
        x = GoogLeNet.Inception(x, 128, name="inception_4a")#512
        x = GoogLeNet.Inception(x, 128, name="inception_4b")
        x = GoogLeNet.Inception(x, 128, name="inception_4c")
        x = GoogLeNet.Inception(x, 132, name="inception_4d")#528
        x = GoogLeNet.Inception(x, 208, name="inception_4e")#832
        """
        x = GoogLeNet.Inception(x, [192,96,208,16,48,64], name="inception_4a_g", trainable=True)#512
        x = GoogLeNet.Inception(x, [160,112,224,24,64,64], name="inception_4b_g", trainable=True)
        x = GoogLeNet.Inception(x, [128,128,256,24,64,64], name="inception_4c_g", trainable=True)
        x = GoogLeNet.Inception(x, [112,144,288,32,64,64], name="inception_4d_g", trainable=True)#528
        x = GoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_4e_g", trainable=True)#832
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        """
        x = GoogLeNet.Inception(x, 208, name="inception_5a")
        x = GoogLeNet.Inception(x, 256, name="inception_5b")#1024
        """
        x = GoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_5a_g", trainable=True)
        x = GoogLeNet.Inception(x, [384,192,384,48,128,128], name="inception_5b_g", trainable=True)#1024
        #x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)
        x = Dense(1024, activation='relu', name="g_1")(x)
        x = Dense(classes, activation='sigmoid', name="g_2")(x)
        # create the model
        model = Model(inpt, x, name='Inception_g')
        # return the constructed network architecture
        if weights == "imagenet":
            print("ImageNet...")
            weights = np.load("/home/anhaoran/codes/par/results/googlenet_weights.npy", encoding='latin1', allow_pickle=True).item()
            for layer in model.layers:
                if layer.get_weights() == []:
                    continue
                #weight = layer.get_weights()
                if layer.name in weights:
                    #print(layer.name, end=':')
                    #print(layer.get_weights()[0].shape == weights[layer.name]['weights'].shape, end=' ')
                    #print(layer.get_weights()[1].shape == weights[layer.name]['biases'].shape)
                    layer.set_weights([weights[layer.name]['weights'], weights[layer.name]['biases']])

        return model
    
    @staticmethod
    def hrp_build(width, height, depth, classes, version="v1", tri_loss=False, center_loss=False, weights="imagenet"):
        inpt = Input(shape=(width, height, depth))
        #padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
        x = GoogLeNet.Conv2d_BN(inpt, 64, (7,7), strides=(2,2), padding='same', name="conv1_7x7_s2", trainable=True)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        x = GoogLeNet.Conv2d_BN(x, 192, (3,3), strides=(1,1), padding='same', name="conv2_3x3", trainable=True)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        """
        x = GoogLeNet.Inception(x, 64, name="inception_3a")#256
        x = GoogLeNet.Inception(x, 120, name="inception_3b")#480
        """
        x = GoogLeNet.Inception(x, [64,96,128,16,32,32], name="inception_3a", trainable=True)#256
        x = GoogLeNet.Inception(x, [128,128,192,32,96,64], name="inception_3b", trainable=True)#480
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        """
        x = GoogLeNet.Inception(x, 128, name="inception_4a")#512
        x = GoogLeNet.Inception(x, 128, name="inception_4b")
        x = GoogLeNet.Inception(x, 128, name="inception_4c")
        x = GoogLeNet.Inception(x, 132, name="inception_4d")#528
        x = GoogLeNet.Inception(x, 208, name="inception_4e")#832
        """
        x = GoogLeNet.Inception(x, [192,96,208,16,48,64], name="inception_4a", trainable=True)#512 
        
        fea_low = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv1_e')(x)
        if len(classes[0]) > 1:
            if version == "v1":
                GoogLeNet.part_attention(fea_low, classes[0], "v1", tri_loss, center_loss, "low")
            elif version == "v2":
                GoogLeNet.part_attention_spatial(fea_low, classes[0], "v2", "low")
            predictions_low =concatenate(GoogLeNet.predictions_low, axis=1)
        else:
            fea_low = GlobalAveragePooling2D()(fea_low)
            predictions_low = Dense(classes[0][0], activation='sigmoid', name="low_fc")(fea_low)
            
        x = GoogLeNet.Inception(x, [160,112,224,24,64,64], name="inception_4b", trainable=True)
        x = GoogLeNet.Inception(x, [128,128,256,24,64,64], name="inception_4c", trainable=True)
        x = GoogLeNet.Inception(x, [112,144,288,32,64,64], name="inception_4d", trainable=True)#528
        
        fea_mid = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv2_e')(x)
        if version == "v1":
            GoogLeNet.part_attention(fea_mid, classes[1], "v1", tri_loss, center_loss, "mid")
        elif version == "v2":
            GoogLeNet.part_attention(fea_mid, classes[1], "v1", tri_loss, center_loss, "mid")
            #GoogLeNet.part_attention_spatial(fea_mid, classes[1], "v2", "mid")
            #GoogLeNet.part_attention_channel(fea_mid, classes[1], "v2", "mid")
        predictions_mid = concatenate(GoogLeNet.predictions_mid, axis=1)
        
        x = GoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_4e", trainable=True)#832
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', trainable=True)(x)
        """
        x = GoogLeNet.Inception(x, 208, name="inception_5a")
        x = GoogLeNet.Inception(x, 256, name="inception_5b")#1024
        """
        x = GoogLeNet.Inception(x, [256,160,320,32,128,128], name="inception_5a", trainable=True)
        x = GoogLeNet.Inception(x, [384,192,384,48,128,128], name="inception_5b", trainable=True)#1024
        #x = AveragePooling2D(pool_size=(7,7), strides=(7,7), padding='same')(x)
        
        fea_hig = Conv2D(1024, (3, 3), padding='same', activation='relu', name='conv3_e')(x)
        if len(classes[2]) > 1:
            if version == "v1":
                GoogLeNet.part_attention(fea_hig, classes[2], "v1", tri_loss, center_loss, "hig")
            elif version == "v2":
                GoogLeNet.part_attention_channel(fea_hig, classes[2], "v2", "hig")
            predictions_hig =concatenate(GoogLeNet.predictions_hig, axis=1)
        else:
            fea_hig = GlobalAveragePooling2D()(fea_hig)
            predictions_hig = Dense(classes[2][0], activation='sigmoid', name="hig_fc")(fea_hig)
        
        predictions_priori = concatenate([predictions_low, predictions_mid], axis=1)
        dim_hig = 0
        for h in classes[2]:
            dim_hig = dim_hig + h
        predictions_hig_cond = Dense(dim_hig, activation="sigmoid", name="high_cond")(predictions_priori)
        predictions_hig_posterior = Lambda(lambda x:x[1] * x[0], name="high_post")([predictions_hig_cond, predictions_hig])
        predictions = concatenate([predictions_low, predictions_mid, predictions_hig_posterior], axis=1)
        #output = concatenate([GoogLeNet.predictions], axis=-1)
        output = predictions
        # create the model
        if tri_loss:
            outputs = [output]
            for i in ws:
                #print(i)
                outputs.append(i)
            oupt = concatenate(outputs, axis=-1)
            model = Model(inpt, oupt, name='Inception')
        elif center_loss:
            outputs = [output, l2_loss]
            oupt = concatenate(outputs, axis=-1)
            model = Model([inpt, input_target], oupt, name='Inception')
        else:
            model = Model(inpt, output, name='Inception')
        # return the constructed network architecture
        if weights == "imagenet":
            print("ImageNet...")
            weights = np.load("/home/anhaoran/codes/spatial_attribute/results/googlenet_weights.npy", encoding='latin1', allow_pickle=True).item()
            for layer in model.layers:
                if layer.get_weights() == []:
                    continue
                #weight = layer.get_weights()
                if layer.name in weights:
                    #print(layer.name, end=':')
                    #print(layer.get_weights()[0].shape == weights[layer.name]['weights'].shape, end=' ')
                    #print(layer.get_weights()[1].shape == weights[layer.name]['biases'].shape)
                    layer.set_weights([weights[layer.name]['weights'], weights[layer.name]['biases']])

        return model
    
def build_orig_inception(nb_classes, width=224, height=224, depth=3, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    model = GoogLeNet.orig_build(width, height, depth, nb_classes)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model

def build_hrp_inception(nb_classes, version="v1", width=299, height=299, depth=3, tri_loss=False, center_loss=False, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]):
    model = GoogLeNet.hrp_build(width, height, depth, nb_classes, version, tri_loss, center_loss)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    
    return model

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    build_hrp_inception([[3, 1], [4, 5, 6, 7, 8], [9, 1]], version="v2")