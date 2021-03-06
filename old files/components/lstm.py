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
num_layers = 4
fillength = 5
nbindex = 128
dropout = 0.15
n_batch = 256
n_epoch = 500

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

prefix = '..//'
h_feature_vector = np.load(prefix + 'Features//h_feature_vector_48.npy')
h_label_vector = np.load(prefix + 'Features//h_label_vector_48.npy')
a_feature_vector = np.load(prefix + 'Features//a_feature_vector_48.npy')
a_label_vector = np.load(prefix + 'Features//a_label_vector_48.npy')
n_feature_vector = np.load(prefix + 'Features//n_feature_vector_48.npy')
n_label_vector = np.load(prefix + 'Features//n_label_vector_48.npy')
s_feature_vector = np.load(prefix + 'Features//s_feature_vector_48.npy')
s_label_vector = np.load(prefix + 'Features//s_label_vector_48.npy')

h_feature_vector_test = np.load(prefix + 'Features//h_feature_vector_test_48.npy')
h_label_vector_test = np.load(prefix + 'Features//h_label_vector_test_48.npy')
a_feature_vector_test = np.load(prefix + 'Features//a_feature_vector_test_48.npy')
a_label_vector_test = np.load(prefix + 'Features//a_label_vector_test_48.npy')
n_feature_vector_test = np.load(prefix + 'Features//n_feature_vector_test_48.npy')
n_label_vector_test = np.load(prefix + 'Features//n_label_vector_test_48.npy')
s_feature_vector_test = np.load(prefix + 'Features//s_feature_vector_test_48.npy')
s_label_vector_test = np.load(prefix + 'Features//s_label_vector_test_48.npy')

h_label_vector[h_label_vector == 0] = 0
a_label_vector[a_label_vector == 1] = 1
h_label_vector_test[h_label_vector_test == 0] = 0
a_label_vector_test[a_label_vector_test == 1] = 1

h_label_vector = to_categorical(h_label_vector, num_classes = 2)
a_label_vector = to_categorical(a_label_vector, num_classes = 2)
h_label_vector_test = to_categorical(h_label_vector_test, num_classes = 2)
a_label_vector_test = to_categorical(a_label_vector_test, num_classes = 2)

# Load training npy files
featureSet_training = np.vstack((h_feature_vector, a_feature_vector))
label_training = np.vstack((h_label_vector, a_label_vector))

# Load testing npy files
featureSet_testing = np.vstack((h_feature_vector_test, a_feature_vector_test))
label_testing = np.vstack((h_label_vector_test, a_label_vector_test))

def float_compatible(input_np):

    x = np.where(input_np >= np.finfo(np.float32).max)
    for index in range(0, len(x[0])):
        x_position = x[0][index]
        y_position = x[1][index]      
        input_np[x_position, y_position] = 0.0
    input_np = np.nan_to_num(input_np)
        
    return input_np

train_data = float_compatible((featureSet_training).astype(np.float32))
eval_data = float_compatible((featureSet_testing).astype(np.float32))

adam = optimizers.Adam(lr = 1e-5, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0, amsgrad = True)
sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
rmsprop = optimizers.RMSprop(lr = 0.0001, rho = 0.9, epsilon = None, decay = 0.0)
adagrad = optimizers.Adagrad(lr = 0.01, epsilon = None, decay = 0.0)
adadelta = optimizers.Adadelta(lr = 1.0, rho = 0.95, epsilon = None, decay = 0.0)
adamax = optimizers.Adamax(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0)
nadam = optimizers.Nadam(lr = 0.002, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, schedule_decay = 0.004)

featureSet = train_data
Label = label_training
featureSet = np.split(featureSet, np.array([NumofFeaturetoUse]), axis = 2)[0]

print('training data: ' + str(featureSet.shape))
print('training label: ' + str(Label.shape))

featureSet_val = eval_data
Label_val = label_testing
featureSet_val = np.split(featureSet_val, np.array([NumofFeaturetoUse]), axis = 2)[0]

print('evaluation data: ' + str(featureSet_val.shape))
print('evaluation label: ' + str(Label_val.shape))

def record(str_message, log_file):
    str_message = str_message + '\n'
    file = open(log_file, 'a')
    file.write(str_message)
    file.close()

def create_lstm(title, num_layers, n_neurons, n_batch, nbindex, dropout, classes, dense_layers):

    model = Sequential()

    model.add(LSTM(nbindex, input_shape=(featureSet.shape[1], featureSet.shape[2]), recurrent_activation='hard_sigmoid',
                   use_bias=True, return_sequences=True))

    model.add(LeakyReLU(alpha=0.05))
    
    model.add(LSTM(nbindex*3, recurrent_activation='hard_sigmoid',
                   use_bias=True, return_sequences=True))

    model.add(LeakyReLU(alpha=0.05))

    model.add(LSTM(nbindex*2, recurrent_activation='hard_sigmoid',
                   use_bias=True, return_sequences=True))

    model.add(Flatten())

    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.summary()

    return model

def train_lstm():
    save_to_path = str(num_layers) + '_Layer(s)_lstm//'

    checkpoint_filepath = str(num_layers) + "_Layer(s)_lstm//Checkpoint_" + title + ".hdf5"
    final_filepath = str(num_layers) + "_Layer(s)_lstm//Final_" + title + ".hdf5"

    if not os.path.exists(save_to_path):
        os.mkdir(save_to_path)

    X, X_test, Y, Y_test= train_test_split(featureSet, Label, test_size = 0.25, shuffle = True)

    model = create_lstm(title, num_layers, n_neurons, n_batch, nbindex, dropout, classes, dense_layers)

    checkpoint = ModelCheckpoint(checkpoint_filepath, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto')

    early_stopping_monitor = EarlyStopping(patience = 5)

    callbacks_list = [checkpoint, early_stopping_monitor]

    model.fit(X, Y, nb_epoch = n_epoch, batch_size = n_batch,  callbacks = callbacks_list, validation_data = (X_test, Y_test), verbose = 1)

    model.save_weights(final_filepath)

    model.load_weights(checkpoint_filepath)

    return model

def predict_lstm(model):
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
    print('Precision: ' + str(precision_score,(y_true, y_pred)))
    print('Recall: ' + str(recall_score(y_true, y_pred)))
    print('f1 score: ' + str(f1_score(y_true, y_pred)))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('true positive ' + str(tp))
    print('false positive ' + str(fp))
    print('false negative ' + str(fn))
    print('true negative ' + str(tn))

title = 'H_A_neurons_' + str(n_neurons) + '_filters_' + str(
    nbindex) + '_dropout_' + str(dropout) + '_epoch_' + str(n_epoch)

final_filepath = str(num_layers) + "_Layer(s)_lstm//Final_" + title + ".hdf5"
#model = load_model(final_filepath)
model = train_lstm()
predict_lstm(model)
