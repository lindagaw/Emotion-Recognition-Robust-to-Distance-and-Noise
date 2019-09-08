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
from keras.models import load_model
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

NumofFeaturetoUse = 100

prefix = '..//..//'
h_feature_vector = np.load(prefix + 'Features//h_feature_vector_48.npy')
h_label_vector = np.load(prefix + 'Features//h_label_vector_48.npy')
a_feature_vector = np.load(prefix + 'Features//a_feature_vector_48.npy')
a_label_vector = np.load(prefix + 'Features//a_label_vector_48.npy')
n_feature_vector = np.load(prefix + 'Features//n_feature_vector_48.npy')
n_label_vector = np.load(prefix + 'Features//n_label_vector_48.npy')
s_feature_vector = np.load(prefix + 'Features//s_feature_vector_48.npy')
s_label_vector = np.load(prefix + 'Features//s_label_vector_48.npy')

h_feature_vector_test = np.load(
    prefix + 'Features//h_feature_vector_test_48.npy')
h_label_vector_test = np.load(prefix + 'Features//h_label_vector_test_48.npy')
a_feature_vector_test = np.load(
    prefix + 'Features//a_feature_vector_test_48.npy')
a_label_vector_test = np.load(prefix + 'Features//a_label_vector_test_48.npy')
n_feature_vector_test = np.load(
    prefix + 'Features//n_feature_vector_test_48.npy')
n_label_vector_test = np.load(prefix + 'Features//n_label_vector_test_48.npy')
s_feature_vector_test = np.load(
    prefix + 'Features//s_feature_vector_test_48.npy')
s_label_vector_test = np.load(prefix + 'Features//s_label_vector_test_48.npy')

h_label_vector[h_label_vector == 0] = 0
a_label_vector[a_label_vector == 1] = 1
h_label_vector_test[h_label_vector_test == 0] = 0
a_label_vector_test[a_label_vector_test == 1] = 1

h_label_vector = to_categorical(h_label_vector, num_classes=2)
a_label_vector = to_categorical(a_label_vector, num_classes=2)
h_label_vector_test = to_categorical(h_label_vector_test, num_classes=2)
a_label_vector_test = to_categorical(a_label_vector_test, num_classes=2)

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

featureSet_val = eval_data
Label_val = label_testing
featureSet_val = np.split(
    featureSet_val, np.array([NumofFeaturetoUse]), axis=2)[0]

print('evaluation data: ' + str(featureSet_val.shape))
print('evaluation label: ' + str(Label_val.shape))


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


final_filepath = prefix + 'modules//Checkpoint_H_A_neurons_4096_filters_512_dropout_0.2_epoch_50000.hdf5'
model = load_model(final_filepath)
predict_cnn(model)
