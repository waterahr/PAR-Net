from keras import backend as K
import tensorflow as tf
import numpy as np

def focal_loss(gamma):

    def loss_interface(y_true, y_pred):
        epsilon = K.epsilon()
        pt = y_pred * y_true + (1-y_pred) * (1-y_true)
        pt = K.clip(pt, epsilon, 1-epsilon)
        CE = -K.log(pt)
        FL = K.pow(1-pt, gamma) * CE
        loss = K.sum(FL, axis=1)
        return loss

    return loss_interface

def weighted_binary_crossentropy(alpha):

    def loss_interface(y_true, y_pred):
        """
        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_pred, y_true)

        # Apply the weights
        weight_vector = y_true * np.exp(-alpha) + (1. - y_true) * 1
        weighted_b_ce = weight_vector * b_ce
        """
        """
        print(y_pred)
        print(1.0-y_pred)
        print(y_true)
        print(1.0-y_true)
        print(K.log(y_pred))
        print(alpha)
        #"""
        """
        Tensor("dense_2/Sigmoid:0", shape=(?, 51), dtype=float32)
        Tensor("loss/dense_2_loss/sub:0", shape=(?, 51), dtype=float32)
        Tensor("dense_2_target:0", shape=(?, ?), dtype=float32)
        Tensor("loss/dense_2_loss/sub_1:0", shape=(?, ?), dtype=float32)
        Tensor("loss/dense_2_loss/Log:0", shape=(?, 51), dtype=float32)
        """
        #b_ce = K.sum(-y_pred * K.log(y_true) - (1.0 - y_pred) * K.log(1.0 - y_true), axis=-1)
        ###NaN
        epsilon = K.epsilon()
        # y_pred = y_pred + epsilon
        """
        logits = K.log(y_pred + epsilon) - K.log(1.0 - y_pred + epsilon)
        b_ce = logits - logits * y_true - K.log(y_pred + epsilon)
        #print(b_ce)#Tensor("loss/dense_2_loss/Sum:0", shape=(?,), dtype=float32)
        weighted_b_ce = logits - logits * y_true - (y_true * alpha + 1 - y_true) * K.log(y_pred + epsilon)
        #"""
        ###exp
        # weighted_b_ce = - y_true * K.log(y_pred + epsilon) * np.exp(alpha) - (1.0 - y_true) * K.log(1.0 - y_pred + epsilon) * np.exp(alpha)
        ###noexp
        weighted_b_ce = -1 / (2 * alpha) * y_true * K.log(y_pred + epsilon) - (1.0 - y_true)/ (2 * (1 - alpha) + epsilon)  * K.log(1.0 - y_pred + epsilon)


        # Return the mean error
        # return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        return K.mean(weighted_b_ce, axis=-1)

    return loss_interface

def mA(y_true, y_pred):
    """
    y_pred_np = K.eval(y_pred)
    y_true_np = K.eval(y_true)
    M = len(y_pred_np)
    L = len(y_pred_np[0])
    res = 0
    for i in range(L):
        P = sum(y_true_np[:, i])
        N = M - P
        TP = sum(y_pred_np[:, i]*y_true_np[:, i])
        TN = list(y_pred_np[:, i]+y_true_np[:, i] == 0.).count(True)
        #print(TP, P, TN, N)
        #print(P,',', N,',', TP,',', TN)
        #if P != 0:
        res += TP/P + TN/N
    return res / (2*L)
    
    y_pred = K.cast(y_pred >= 0.5, dtype='float32')
    y_true = K.cast(y_true >= 0.5, dtype='float32')
    #print(K.int_shape(y_true))
    P = K.sum(y_true, axis=-1) + K.epsilon()
    #print("P", P)
    N = K.sum(1-y_true, axis=-1) + K.epsilon()
    #print("N", N)
    TP = K.sum(y_pred * y_true, axis=-1)
    #print("TP", TP)
    TN = K.sum(K.cast_to_floatx(y_pred + y_true == 0))
    #print("TN", TN)
    return K.mean(TP / P + TN / N) / 2"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=-1)
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)), axis=-1)
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)), axis=-1)
    mean_acc = (true_positives / (possible_positives + K.epsilon()) + true_negatives / (possible_negatives + K.epsilon())) / 2
    return mean_acc

def center_weighted_binary_crossentropy(alpha, nb_class):

    def loss_interface(y_true, y_out):
        print(y_true)#Tensor("concatenate_10_target:0", shape=(?, ?), dtype=float32)
        print(y_out)#Tensor("concatenate_10/concat:0", shape=(?, 51), dtype=float32)
        print(y_out[:, nb_class:])
        epsilon = K.epsilon()
        y_pred = y_out[:, :nb_class]
        #print(y_pred)
        weighted_b_ce = -1 / (2 * alpha) * y_true * K.log(y_pred + epsilon) - (1.0 - y_true)/ (2 * (1 - alpha))  * K.log(1.0 - y_pred + epsilon)
        
        return K.mean(weighted_b_ce, axis=-1) + 0.2 * K.mean(y_out[:, nb_class:], axis=-1)

    return loss_interface

def tri_weighted_binary_crossentropy(alpha, nb_class):

    def loss_interface(y_true, y_out):
        print(y_true)#Tensor("concatenate_10_target:0", shape=(?, ?), dtype=float32)
        print(y_out)#Tensor("concatenate_10/concat:0", shape=(?, 51), dtype=float32)
        epsilon = K.epsilon()
        y_pred = y_out[:, :nb_class]
        #print(y_pred)
        weighted_b_ce = -1 / (2 * alpha) * y_true * K.log(y_pred + epsilon) - (1.0 - y_true)/ (2 * (1 - alpha))  * K.log(1.0 - y_pred + epsilon)
        
        whole, hs, ub, lb, sh, at = y_out[:, nb_class:1024+51], y_out[:, 1024+nb_class:1024*2+nb_class], y_out[:, 1024*2+nb_class:1024*3+nb_class], y_out[:, 1024*3+nb_class:1024*4+nb_class], y_out[:, 1024*4+nb_class:1024*5+nb_class], y_out[:, 1024*5+nb_class:]
        #other all???
        #"""
        print(whole, hs, ub, lb, sh, at)
        print(tf.random_shuffle(hs, seed=1))
        #"""
        #"""
        dis_pos = K.sum(K.square(hs - hs[0]), axis=1, keepdims=True)
        dis_pos += K.sum(K.square(ub - ub[0]), axis=1, keepdims=True)
        dis_pos += K.sum(K.square(lb - lb[0]), axis=1, keepdims=True)
        dis_pos += K.sum(K.square(sh - sh[0]), axis=1, keepdims=True)
        dis_pos = K.sqrt(dis_pos)
        #"""
        dis_neg = K.sum(K.square(hs - ub), axis=1, keepdims=True)
        dis_neg += K.sum(K.square(hs - lb), axis=1, keepdims=True)
        dis_neg += K.sum(K.square(hs - sh), axis=1, keepdims=True)
        dis_neg += K.sum(K.square(ub - lb), axis=1, keepdims=True)
        dis_neg += K.sum(K.square(ub - sh), axis=1, keepdims=True)
        dis_neg += K.sum(K.square(lb - sh), axis=1, keepdims=True)
        dis_neg = K.sqrt(dis_neg)
        a1 = 1000###800--->remove dis_pos
        d = K.maximum(0.0, dis_pos - dis_neg + a1)

        return K.mean(weighted_b_ce, axis=-1) + 0.2 * K.mean(d, axis=-1)

    return loss_interface

def tri_mA(nb_class):
    def mA(y_true, y_out):
        y_pred = y_out[:, :nb_class]
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=-1)
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=-1)
        true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)), axis=-1)
        possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)), axis=-1)
        mean_acc = (true_positives / (possible_positives + K.epsilon()) + true_negatives / (possible_negatives + K.epsilon())) / 2
        return mean_acc
    return mA