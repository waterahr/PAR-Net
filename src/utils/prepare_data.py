import numpy as np
import pandas as pd
import random
import scipy.io as sio
import os
import h5py
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

def generate_image_from_nmlist(Xnms, y, batch_size=64, image_shape=(299, 299)):
    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True)
    while True:
        cnt = 0
        X = []
        Y = []
        indices = np.arange(len(Xnms))
        random.shuffle(indices)
        Xnms = Xnms[list(indices)]
        y = y[list(indices)]
        for i in range(len(Xnms)):
            img = image.load_img(Xnms[i], target_size=(image_shape[0], image_shape[1], 3))
            img = image.img_to_array(img)
            X.append(img)
            Y.append(y[i])
            cnt += 1
            if cnt==batch_size:
                X = np.asarray(X)
                Y = np.asarray(Y)
                datagen.fit(X)
                yield (X, Y)
                cnt = 0
                X = []
                Y = []
                
def generate_imageandtarget_from_nmlist(Xnms, y, batch_size=64, image_shape=(299, 299)):
    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True)
    while True:
        cnt = 0
        X = []
        X_ = np.tile(np.arange(0, 6), (batch_size, 1))
        Y = []
        indices = np.arange(len(Xnms))
        random.shuffle(indices)
        Xnms = Xnms[list(indices)]
        y = y[list(indices)]
        for i in range(len(Xnms)):
            img = image.load_img(Xnms[i], target_size=(image_shape[0], image_shape[1], 3))
            img = image.img_to_array(img)
            X.append(img)
            Y.append(y[i])
            cnt += 1
            if cnt==batch_size:
                X = np.asarray(X)
                Y = np.asarray(Y)
                datagen.fit(X)
                yield ([X, X_], Y)
                cnt = 0
                X = []
                Y = []
                
def generate_rap(partion=0):
    data_root = "/home/anhaoran/data/pedestrian_attributes_RAP/"
    data = sio.loadmat(data_root + "RAP_annotation/RAP_annotation.mat")["RAP_annotation"]
    
    attributes_list = []
    for i in range(data["attribute_eng"][0][0].shape[0]):
        attributes_list.append(data["attribute_eng"][0][0][i][0][0])
        
    X_data = []
    y_data = []
    for i in range(41585):
        X_data.append(os.path.join(data_root + "RAP_dataset", data['imagesname'][0][0][i][0][0]))
        y_data.append(data['label'][0][0][i])
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)
    
    train_indices, test_indices = data['partion'][0][0][partion][0][0][0]
    train_indices, test_indices = list(train_indices[0] - 1), list(test_indices[0] - 1)
    
    print("=========================================")
    print(attributes_list)
    print("The shape of the X_train is: ", X_data[train_indices].shape)
    print("The shape of the y_train is: ", y_data[train_indices].shape)
    print("The shape of the X_test is: ", X_data[test_indices].shape)
    print("The shape of the y_test is: ", y_data[test_indices].shape)
    print("=========================================")
    return X_data[train_indices], y_data[train_indices], X_data[test_indices], y_data[test_indices], attributes_list

def generate_peta():
    data_root = "/home/anhaoran/codes/pedestrian_attibutes_wpal/results/"
    data = np.array(pd.read_csv(data_root + "PETA.csv"))[:, 1:]
    
    labels_list_data = open("/home/anhaoran/data/pedestrian_attributes_PETA/PETA/labels.txt")
    lines = labels_list_data.readlines()
    attributes_list = []
    for line in lines:
        tmp = line.split()
        attributes_list.append(tmp[1])
        
    X_data = []
    y_data = []
    for i in range(len(data)):
        X_data.append(data[i, 0])
        y_data.append(data[i, 1:])
    X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)
    
    # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33, random_state=42)
    
    # return X_train, y_train, X_test, y_test, attributes_list
    print("=========================================")
    print(attributes_list)
    print("The shape of the X_train is: ", X_data[:11400].shape)
    print("The shape of the y_train is: ", y_data[:11400].shape)
    print("The shape of the X_test is: ", X_data[11400:].shape)
    print("The shape of the y_test is: ", y_data[11400:].shape)
    print("=========================================")
    return X_data[:11400], y_data[:11400], X_data[11400:], y_data[11400:], attributes_list

def generate_pa100k():
    data_root = "/home/anhaoran/data/pedestrian_attributes_PA-100K/"
    data = sio.loadmat(data_root + "annotation/annotation.mat")
    
    attributes_list = []
    for i in range(len(data["attributes"])):
        attributes_list.append(data["attributes"][i][0][0])
        
    X_train = []
    y_train = data['train_label']
    for i in range(len(data['train_images_name'])):
        X_train.append(data_root + "data/release_data/release_data/" + str(data['train_images_name'][i][0][0]))
    X_train = np.asarray(X_train)
    X_val = []
    y_val = data['val_label']
    for i in range(len(data['val_images_name'])):
        X_val.append(data_root + "data/release_data/release_data/" + str(data['val_images_name'][i][0][0]))
    X_val = np.asarray(X_val)
    X_test = []
    y_test = data['test_label']
    for i in range(len(data['test_images_name'])):
        X_test.append(data_root + "data/release_data/release_data/" + str(data['test_images_name'][i][0][0]))
    X_test = np.asarray(X_test)
    
    print("=========================================")
    print(attributes_list)
    print("The shape of the X_train is: ", X_train.shape)
    print("The shape of the y_train is: ", y_train.shape)
    print("The shape of the X_val is: ", X_val.shape)
    print("The shape of the y_val is: ", y_val.shape)
    print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    print("=========================================")
    return X_train, y_train, X_val, y_val, X_test, y_test, attributes_list

def generate_parse27k():
    data_root = "/home/anhaoran/data/pedestrian_attributes_parse-27k/"
    data = h5py.File(data_root + "train.hdf5", 'r')
    X_train = data["crops"][:].transpose((0,2,3,1)).shape
    y_train = data['labels'][:]
    data = h5py.File(data_root + "val.hdf5", 'r')
    X_val = data["crops"][:].transpose((0,2,3,1)).shape
    y_val = data['labels'][:]
    data = h5py.File(data_root + "test.hdf5", 'r')
    X_test = data["crops"][:].transpose((0,2,3,1)).shape
    y_test = data['labels'][:]
    
    print("=========================================")
    print("The shape of the X_train is: ", X_train.shape)
    print("The shape of the y_train is: ", y_train.shape)
    print("The shape of the X_val is: ", X_val.shape)
    print("The shape of the y_val is: ", y_val.shape)
    print("The shape of the X_test is: ", X_test.shape)
    print("The shape of the y_test is: ", y_test.shape)
    print("=========================================")
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    generate_rap()
    #generate_peta()
    #generate_pa100k()