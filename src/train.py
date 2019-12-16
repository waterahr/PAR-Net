import os
import numpy as np
import glob
import argparse
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger
from keras.optimizers import *

from utils.prepare_data import *
from inception.build_models import *
from utils.train_utils import *

def parse_arg():
    model_nms = ["Inception", "HRPInception"]
    data_nms = ["PETA", "RAP", "PA100K"]
    parser = argparse.ArgumentParser(description='For the training and testing of the model...')
    parser.add_argument('-m', '--model', type=str, default="",
                        help='The model name in ' + str(model_nms) + '.')
    parser.add_argument('-s', '--save', type=str, default="",
                        help='The model savename.')###"center/triplet_v1"
    parser.add_argument('-g', '--gpus', type=str, default="",
                        help='The gpu device\'s ID need to be used.')
    parser.add_argument('-d', '--data', type=str, default="",
                        help='The dataset need to be trained.')
    parser.add_argument('-w', '--weight', type=str, default="",
                        help='The initial weight.')
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='The epochs need to be trained')
    parser.add_argument('-b', '--batch', type=int, default=32,
                        help='The batch size in the training progress.')
    parser.add_argument('-c', '--classes', type=int, default=0,
                        help='The class number.')
    parser.add_argument('-i', '--iteration', type=int, default=0,
                        help='The iteration number.')
    args = parser.parse_args()
    if args.model == "" or args.model not in model_nms:
        raise RuntimeError('NO MODEL FOUND IN ' + str(model_nms))
    if args.data == "" or args.data not in data_nms:
        raise RuntimeError('NO DATABASE FOUND IN ' + str(data_nms))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    return args

if __name__ == "__main__":
    print("-----------------training begining---------------------")
    args = parse_arg()
    model_prefix = "../models/" + args.data + "/" + args.model + "/"
    os.makedirs(model_prefix, exist_ok=True)
    if args.save != "":
        model_prefix = model_prefix + args.save + "_"
    nb_epoch = args.epochs
    nb_class = args.classes
    batch_size = args.batch
    monitor = 'val_mA'
    
    if args.data == "PETA":
        X_train, y_train, X_test, y_test, _ = generate_peta()
    elif args.data == "RAP":
        X_train, y_train, X_test, y_test, _ = generate_rap()
    elif args.data == "PA100K":
        X_train, y_train, X_test, y_test, _, _, _ = generate_pa100k()
      
    if args.data == "PETA":
        if args.save.startswith("coarse"):
            
            idx_indices = list(np.hstack((whole, up, lb))[:nb_class])
        else:
            low = [27,32,50,56]
            hs = [0,8,20,21,25,28,36,37,44,54]
            ub = [23,30,39,46,51,55,58,59,60]
            lb = [22,24,29,45,47,53,57]
            sh = [9,26,42,43,48,49]
            at = [6,7,11,12,13,17,33,35,38,41,52]
            high_whole = [1,2,3,4,5,16,34]
            high_ub = [15,19,40]
            high_lb = [10,14,18,31]
            parts = [[len(low)], [len(hs), len(ub), len(lb), len(sh), len(at)], [len(high_whole), len(high_ub), len(high_lb)]]
            idx_indices = list(np.hstack((low, hs, ub, lb, sh, at, high_whole, high_ub, high_lb)))
    elif args.data == "RAP":
        if args.save.startswith("coarse"):
            
            idx_indices = list(np.hstack((whole, up, lb))[:nb_class])
        else:
            low = [11]
            hs = [9,10,12,13,14]
            ub = [15,16,17,18,19,20,21,22,23]
            lb = [24,25,26,27,28,29]
            sh = [30,31,32,33,34]
            at = [35,36,37,38,39,40,41,42]
            high = [0,1,2,3,4,5,6,7,8,43,44,45,46,47,48,49,50]
            parts = [[len(low)], [len(hs), len(ub), len(lb), len(sh), len(at)], [len(high)]]
            idx_indices = list(np.hstack((low, hs, ub, lb, sh, at, high)))
    elif args.data == "PA100K":
        if args.save.startswith("coarse"):
            
            idx_indices = list(np.hstack((whole, up, lb))[:nb_class])
        else:
            
            idx_indices = list(np.hstack((whole, hs, ub, lb, sh, at))[:nb_class])
    if args.model == "HRPInception":
        y_train = y_train[:, idx_indices]
        y_test = y_test[:, idx_indices]
    else:
        y_train = y_train[:, :nb_class]
        y_test = y_test[:, :nb_class]
    
    #loss_func = ""
    #loss_weights = None
    #metrics=[]
    center_loss = False
    alpha = np.sum(y_train, axis=0)#(len(data_y[0]), )
    alpha = alpha / len(y_train)
    print(alpha)
    image_shape=(299, 299)
    
    opt_sgd = "adam"
    #opt_sgd = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #opt_sgd = RMSprop(lr=0.001, rho=0.9)
    #opt_sgd = Adagrad(lr=0.01)###***
    opt_sgd = Adadelta(lr=1.0, rho=0.95)###*****
    #opt_sgd = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999)###*****
    #opt_sgd = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)###***
    #opt_sgd=SGD(lr=0.0001, momentum=0.9,decay=0.0001,nesterov=True)
    #model_prefix = model_prefix + "sgd_"
    
    #version=args.save, 
    if args.model == "Inception":
        image_shape=(224, 224)
        model = build_orig_inception(nb_class, metrics=["accuracy", mA], optimizer=opt_sgd)
    elif args.model == "HRPInception":
        if args.save.startswith("v"):
            loss_func = weighted_binary_crossentropy(alpha[:nb_class])
            metric_lis = ["accuracy", mA]
            tri_loss = False
            center_loss = False
        elif args.save.startswith("triplet"):
            loss_func = tri_weighted_binary_crossentropy(alpha[:nb_class], nb_class)
            metric_lis = [tri_mA(nb_class)]
            tri_loss = True
            center_loss = False
        elif args.save.startswith("center"):
            loss_func = center_weighted_binary_crossentropy(alpha[:nb_class], nb_class)
            metric_lis = [tri_mA(nb_class)]
            tri_loss = False
            center_loss = True
        model = build_hrp_inception(parts, version=args.save[args.save.index("v"):], 
            width=image_shape[1], height=image_shape[0],
            tri_loss = tri_loss, center_loss = center_loss, loss=loss_func, metrics=metric_lis, optimizer=opt_sgd)
    
    if args.weight != "":
        model.load_weights(args.weight, by_name=True)
        
    train_generator = generate_image_from_nmlist(X_train, y_train, batch_size, image_shape)
    val_generator = generate_image_from_nmlist(X_test, y_test, batch_size, image_shape)
    if center_loss:
        train_generator = generate_imageandtarget_from_nmlist(X_train, y_train, batch_size, image_shape)
        val_generator = generate_imageandtarget_from_nmlist(X_test, y_test, batch_size, image_shape)
    checkpointer = ModelCheckpoint(filepath = model_prefix + 'epoch{epoch:03d}_valloss{'+ monitor + ':.6f}.hdf5',
                        monitor = monitor,
                        verbose=1, 
                        save_best_only=True, 
                        save_weights_only=True,
                        mode='max',#'auto', 
                        period=1)
    csvlog = CSVLogger(model_prefix + str(args.epochs) + 'iter' + '_log.csv', append=True)
    def step_decay(epoch):
        initial_lrate = 0.001
        gamma = 0.75
        step_size = 200
        lrate = initial_lrate * math.pow(gamma, math.floor((1+epoch) / step_size))
        return lrate
    lrate = LearningRateScheduler(step_decay)
    model.fit_generator(train_generator,
            steps_per_epoch = int(X_train.shape[0] / batch_size),
            epochs = nb_epoch,
            validation_data = val_generator,
            validation_steps = int(X_test.shape[0] / batch_size),
            callbacks = [checkpointer, csvlog], #, lrate
            workers = 1,
            initial_epoch = args.iteration)#
    model.save_weights(model_prefix + 'final' + str(args.epochs) + 'iter_model.h5')
    print("-----------------training endding---------------------")