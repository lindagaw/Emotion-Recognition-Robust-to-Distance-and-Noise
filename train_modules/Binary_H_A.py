import random
import os
import shutil
import glob
import gc
import sys
import h5py
import time
import datetime
import pickle
import librosa
import warnings
import matplotlib.pyplot as plt

import numpy as np
from numpy import array
import pandas as pd
from pandas.plotting import parallel_coordinates
from pydub import AudioSegment

#imported for testing
import wave
import contextlib

# for outputing file
from scipy.cluster.vq import vq, kmeans, whiten
import scipy.stats.stats as st

from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize
from sklearn.metrics import auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn import mixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import class_weight

import keras

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten, Add, Dropout, Input, Activation
from keras.layers import TimeDistributed, Bidirectional, LSTM, LeakyReLU
from keras.models import Sequential
from keras import optimizers, regularizers
from keras.utils import np_utils, to_categorical

from colorama import Fore, Back, Style

from IPython.display import clear_output


#warnings.filterwarnings('ignore')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# confirm TensorFlow sees the GPU
from tensorflow.python.client import device_lib
# assert 'GPU' in str(device_lib.list_local_devices())

# confirm Keras sees the GPU
from keras import backend
# print(len(backend.tensorflow_backend._get_available_gpus()) > 0)

#warnings.filterwarnings('ignore')
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(1, '..//components//')
import load_feat_directories

sample_rate = 44100
frame_number = 48
hop_length = 441  # frame size= 2 * hop
segment_length = int(sample_rate * 0.2)  # 0.2
segment_pad = int(sample_rate * 0.02)     # 0.02
overlapping = int(sample_rate * 0.1)   # 0.1

classes = 2
NumofFeaturetoUse = 272
n_neurons = 4096
dense_layers = 1
num_layers = 3
fillength = 4
nbindex = 512
dropout = 0.2
n_batch = 128
n_epoch = 1000

def update_progress(progress):
    bar_length = 100
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))
    clear_output(wait = True)
    
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

# allnoised_npy[0, 1, 2, 3, 4] ==> H, A, N, S, O
# homenoised_npy[0, 1, 2, 3, 4] ==> H, A, N, S, O
all_noised_npy = load_feat_directories.allnoised_npy
all_noised_npy_test = load_feat_directories.allnoised_npy_test
home_noised_npy = load_feat_directories.homenoised_npy
home_noised_npy_test = load_feat_directories.homenoised_npy_test


for index in range(0, 5):
    #x = os.path.exists(all_noised_npy[index])
    #y = os.path.exists(home_noised_npy[index])

    if not os.path.exists(all_noised_npy[index]):
        print(all_noised_npy[index] + ' does not exist. Breaking the loop... ')

    if not os.path.exists(home_noised_npy[index]):
        print(home_noised_npy[index] + 'does not exist. Breaking the loop... ')


def comprise_vector(path):
    vec_to_return = np.array([])
    for fname in os.listdir(path):
        if not fname.endswith('.npy'):
            continue
        print(fname)
        current_vec = np.load(path + fname)
        if len(list(vec_to_return)) == 0:
            vec_to_return = current_vec
        else:
            vec_to_return = np.vstack((vec_to_return, current_vec))
        if len(list(vec_to_return)) == 2:
            break
    return vec_to_return

def comprise_label(feature_vector, label):
    length = len(list(feature_vector))
    label_vec_to_ret  = np.array([])
    for i in range(0, length):
        current = np.array([label])
        if len(list(label_vec_to_ret)) == 0:
            label_vec_to_ret = current
        else:
            label_vec_to_ret = np.vstack((label_vec_to_ret, current))
        if len(list(label_vec_to_ret)) == 2:
            break
    return label_vec_to_ret


for index in [0, 1]:
    if not os.path.exists(all_noised_npy[index]):
        print(all_noised_npy[index] + ' does not exist.')
    else:
        path = all_noised_npy[index]
        if index == 0:
            h_feature_vector_all = comprise_vector(path)
            h_label_vector_all = comprise_label(h_feature_vector_all, index)
        elif index == 1:
            a_feature_vector_all = comprise_vector(path)
            a_label_vector_all = comprise_label(a_feature_vector_all, index)
        elif index == 2:
            n_feature_vector_all = comprise_vector(path)
            n_label_vector_all = comprise_label(n_feature_vector_all, index)
        elif index == 3:
            s_feature_vector_all = comprise_vector(path)
            s_label_vector_all = comprise_label(s_feature_vector_all, index)
        else:
            o_feature_vector_all = comprise_vector(path)
            o_label_vector_all = comprise_label(o_feature_vector_all, index)
    '''
    if not os.path.exists(home_noised_npy[index]):
        print(home_noised_npy[index] + 'does not exist.')
    else:
        path = home_noised_npy[index]
        if index == 0:
            h_feature_vector_home = comprise_vector(path)
            h_label_vector_home = comprise_label(h_feature_vector_home, index)
        elif index == 1:
            a_feature_vector_home = comprise_vector(path)
            a_label_vector_home = comprise_label(a_feature_vector_home, index)
        elif index == 2:
            n_feature_vector_home = comprise_vector(path)
            n_label_vector_home = comprise_label(n_feature_vector_home, index)
        elif index == 3:
            s_feature_vector_home = comprise_vector(path)
            s_label_vector_home = comprise_label(s_feature_vector_home, index)
        else:
            o_feature_vector_home = comprise_vector(path)
            o_label_vector_home = comprise_label(o_feature_vector_home, index)
    '''

for index in [0, 1]:

    if not os.path.exists(all_noised_npy_test[index]):
        print(all_noised_npy_test[index] + ' does not exist.')
    else:
        path = all_noised_npy_test[index]
        if index == 0:
            h_feature_vector_all_test = comprise_vector(path)
            h_label_vector_all_test = comprise_label(h_feature_vector_all_test, index)
        elif index == 1:
            a_feature_vector_all_test = comprise_vector(path)
            a_label_vector_all_test = comprise_label(a_feature_vector_all_test, index)
        elif index == 2:
            n_feature_vector_all_test = comprise_vector(path)
            n_label_vector_all_test = comprise_label(n_feature_vector_all_test, index)
        elif index == 3:
            s_feature_vector_all_test = comprise_vector(path)
            s_label_vector_all_test = comprise_label(s_feature_vector_all_test, index)
        else:
            o_feature_vector_all_test = comprise_vector(path)
            o_label_vector_all_test = comprise_label(o_feature_vector_all_test, index)

    '''
    if not os.path.exists(home_noised_npy_test[index]):
        print(home_noised_npy_test[index] + 'does not exist.')
    else:
        path = home_noised_npy_test[index]
        if index == 0:
            h_feature_vector_home_test = comprise_vector(path)
            h_label_vector_home_test = comprise_label(h_feature_vector_home_test, index)
        elif index == 1:
            a_feature_vector_home_test = comprise_vector(path)
            a_label_vector_home_test = comprise_label(a_feature_vector_home_test, index)
        elif index == 2:
            n_feature_vector_home_test = comprise_vector(path)
            n_label_vector_home_test = comprise_label(n_feature_vector_home_test, index)
        elif index == 3:
            s_feature_vector_home_test = comprise_vector(path)
            s_label_vector_home_test = comprise_label(s_feature_vector_home_test, index)
        else:
            o_feature_vector_home_test = comprise_vector(path)
            o_label_vector_home_test = comprise_label(o_feature_vector_home_test, index)
    '''

def float_compatible(input_np):

    x = np.where(input_np >= np.finfo(np.float32).max)
    for index in range(0, len(x[0])):
        x_position = x[0][index]
        y_position = x[1][index]
        input_np[x_position, y_position] = 0.0
    input_np = np.nan_to_num(input_np)

    return input_np

# Load training npy files
featureSet = float_compatible(np.vstack((h_feature_vector_all, a_feature_vector_all)))
Label = np.vstack((h_label_vector_all, a_label_vector_all))
#featureSet = np.split(featureSet, np.array([NumofFeaturetoUse]), axis = 2)[0]
print('training data: ' + str(featureSet.shape))
print('training label: ' + str(Label.shape))


# Load testing npy files
featureSet_val = float_compatible(np.vstack((h_feature_vector_all_test, a_feature_vector_all_test)))
Label_val = np.vstack((h_label_vector_all_test, a_label_vector_all_test))
#featureSet_val = np.split(featureSet_val, np.array([NumofFeaturetoUse]), axis = 2)[0]
print('evaluation data: ' + str(featureSet_val.shape))
print('evaluation label: ' + str(Label_val.shape))

adam = optimizers.Adam(lr = 3e-6, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0, amsgrad = True)
sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
rmsprop = optimizers.RMSprop(lr = 0.0001, rho = 0.9, epsilon = None, decay = 0.0)
adagrad = optimizers.Adagrad(lr = 0.01, epsilon = None, decay = 0.0)
adadelta = optimizers.Adadelta(lr = 1.0, rho = 0.95, epsilon = None, decay = 0.0)
adamax = optimizers.Adamax(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0)
nadam = optimizers.Nadam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, schedule_decay = 0.004)


def record(str_message, log_file):
    str_message = str_message + '\n'
    file = open(log_file, 'a')
    file.write(str_message)
    file.close()

def create_cnn(title, num_layers, n_neurons, n_batch, nbindex, dropout, classes, dense_layers):

    model = Sequential()

    model.add(Convolution1D(nb_filter=nbindex, filter_length=fillength,
                            input_shape=(featureSet.shape[1], featureSet.shape[2]), kernel_constraint=maxnorm(3)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(dropout))

    model.add(Convolution1D(nb_filter=nbindex*2, filter_length=fillength,
                            kernel_constraint=maxnorm(3)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(dropout))

    model.add(Convolution1D(nb_filter=nbindex*3, filter_length=fillength,
                            kernel_constraint=maxnorm(3)))
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(dropout))

    model.add(Convolution1D(nb_filter=nbindex*2, filter_length=fillength,
                            kernel_constraint=maxnorm(3)))  
    model.add(LeakyReLU(alpha=0.05))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=[metrics.categorical_accuracy])

    model.summary()

    return model

def train_cnn(prefix):
    
    save_to_path = prefix + str(num_layers) + "_Layer(s)//"

    checkpoint_filepath = prefix + str(num_layers) + "_Layer(s)//Checkpoint_" + title + ".hdf5"
    final_filepath = prefix + str(num_layers) + "_Layer(s)//Final_" + title + ".hdf5"

    if not os.path.exists(save_to_path):
        os.mkdir(save_to_path)

    X, X_test, Y, Y_test= train_test_split(featureSet, Label, test_size = 0.25, shuffle = True)
    model = create_cnn(title, num_layers, n_neurons, n_batch, nbindex, dropout, classes, dense_layers)
    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor = 'val_acc', verbose = 0, save_best_only = True, mode = 'auto')
    early_stopping_monitor = EarlyStopping(patience = 100)
    callbacks_list = [checkpoint, early_stopping_monitor]
    model.fit(X, Y, nb_epoch = n_epoch, batch_size = n_batch,  callbacks = callbacks_list, validation_data = (X_test, Y_test), verbose = 1)
    model.save_weights(final_filepath)
    model.load_weights(checkpoint_filepath)
    return model

def predict_cnn(model):
    y_pred = []
    y_true = []

    for item in list(Label_val):
            if item[0] > item[1]:
                y_true.append(0)
            elif item[0] < item[1]:
                y_true.append(1)
            else:
                y_true.append(0)

    for item in list(model.predict(featureSet_val)):
            if item[0] > item[1]:
                y_pred.append(0)
            elif item[0] < item[1]:
                y_pred.append(1)
            else:
                y_pred.append(0)

    print('Accuracy: ' + str(accuracy_score(y_true, y_pred)))
    print('Precision: ' + str(precision_score(y_true, y_pred)))
    print('Recall: ' + str(recall_score(y_true, y_pred)))
    print('f1 score: ' + str(f1_score(y_true, y_pred)))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('true positive ' + str(tp))
    print('false positive ' + str(fp))
    print('false negative ' + str(fn))
    print('true negative ' + str(tn))

title = 'H_A_neurons_' + str(n_neurons) + '_filters_' + str(nbindex) + '_dropout_' + str(dropout) + '_epoch_' + str(n_epoch)

prefix = '..//..//modules//'
final_filepath = prefix + str(num_layers) + "_Layer(s)//Final_" + title + ".hdf5"

#model = load_model(final_filepath)
#model = train_cnn('..//..//modules//')
#predict_cnn(model)
