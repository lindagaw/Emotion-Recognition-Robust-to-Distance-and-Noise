import init

import random
import time
import datetime
import numpy as np
from numpy import array
import pandas as pd
from pydub import AudioSegment
import os
import shutil
import glob
import gc
import sys
import h5py
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

#imported for testing
import wave
import contextlib

# for outputing file
import scipy.stats.stats as st

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn import mixture
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.impute import SimpleImputer
import pickle
import librosa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from pandas.plotting import parallel_coordinates
import pickle

import keras
from keras import optimizers, regularizers
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.constraints import maxnorm
from keras.layers import Add
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Input
from keras.callbacks import EarlyStopping
from keras.layers import Activation
from keras.utils import np_utils
from keras.utils import to_categorical

from colorama import Fore, Back, Style

from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sample_rate = init.sample_rate
frame_number = init.frame_number
hop_length = init.hop_length
segment_length = init.segment_length
segment_pad = init.segment_pad
overlapping = init.overlapping
classes = init.classes
NumofFeaturetoUse = init.NumofFeaturetoUse
n_neurons = init.n_neurons
dense_layers = init.dense_layers
num_layers = init.num_layers
fillength = init.fillength
nbindex = init.nbindex
dropout = init.dropout
n_batch = init.n_batch
n_epoch = init.n_epoch

model = Sequential()

model.add(Convolution1D(nb_filter=nbindex, filter_length=fillength, activation='relu',
                        input_shape=(featureSet.shape[1], featureSet.shape[2]), kernel_constraint=maxnorm(3)))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
model.add(Dropout(dropout))

for layer in range(0, num_layers-1):
    model.add(Convolution1D(nb_filter=nbindex, filter_length=fillength,
                            activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(Dropout(dropout))

model.add(Flatten())

for layer in range(0, dense_layers):
    model.add(Dense(n_neurons, activation='relu'))
    model.add(Dropout(dropout))

model.add(Dense(classes, activation='softmax'))

model.compile(loss='binary_crossentropy',
                optimizer=adam, metrics=['accuracy'])

model.summary()

