import random
import time
import datetime
import numpy as np
from numpy import array
import pandas as pd
from pydub import AudioSegment
import os, shutil, glob
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
from keras.layers import Dense, Add
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Input
from keras.callbacks import EarlyStopping
from keras.layers import Activation
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.layers import LeakyReLU
from keras.models import load_model

from colorama import Fore, Back, Style

from IPython.display import clear_output

import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sample_rate = 44100
frame_number = 48
hop_length = 441  # frame size= 2 * hop
segment_length = int(sample_rate * 0.2)  #0.2
segment_pad = int(sample_rate * 0.02)     #0.02
overlapping = int(sample_rate * 0.1)   #0.1
fillength = 4  # 6  filer size
classes = 2
NumofFeaturetoUse = 272  # int(sys.argv[1])


classes = 2
NumofFeaturetoUse = 100
n_neurons = 4096
dense_layers = 1
num_layers = 4
fillength = 4
nbindex = 72
dropout = 0.2
n_batch = 128
n_epoch = 10000
# In[2]:


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

h_feature_vector = np.load('..//Features//h_feature_vector_48.npy')
h_label_vector = np.load('..//Features//h_label_vector_48.npy')

a_feature_vector = np.load('..//Features//a_feature_vector_48.npy')
a_label_vector = np.load('..//Features//a_label_vector_48.npy')

n_feature_vector = np.load('..//Features//n_feature_vector_48.npy')
n_label_vector = np.load('..//Features//n_label_vector_48.npy')

s_feature_vector = np.load('..//Features//s_feature_vector_48.npy')
s_label_vector = np.load('..//Features//s_label_vector_48.npy')

h_feature_vector_test = np.load('..//Features//h_feature_vector_test_48.npy')
h_label_vector_test = np.load('..//Features//h_label_vector_test_48.npy')

a_feature_vector_test = np.load('..//Features//a_feature_vector_test_48.npy')
a_label_vector_test = np.load('..//Features//a_label_vector_test_48.npy')

n_feature_vector_test = np.load('..//Features//n_feature_vector_test_48.npy')
n_label_vector_test = np.load('..//Features//n_label_vector_test_48.npy')

s_feature_vector_test = np.load('..//Features//s_feature_vector_test_48.npy')
s_label_vector_test = np.load('..//Features//s_label_vector_test_48.npy')

h_label_vector[h_label_vector == 0] = 0
a_label_vector[a_label_vector == 1] = 1

h_label_vector_test[h_label_vector_test == 0] = 0
a_label_vector_test[a_label_vector_test == 1] = 1


# In[6]:
h_label_vector = to_categorical(h_label_vector, num_classes=2)
a_label_vector = to_categorical(a_label_vector, num_classes=2)

h_label_vector_test = to_categorical(h_label_vector_test, num_classes=2)
a_label_vector_test = to_categorical(a_label_vector_test, num_classes=2)


# In[7]:


# Load training npy files
featureSet_training = np.vstack((h_feature_vector, a_feature_vector))
label_training = np.vstack((h_label_vector, a_label_vector))

# Load testing npy files
featureSet_testing = np.vstack((h_feature_vector_test, a_feature_vector_test))
label_testing = np.vstack((h_label_vector_test, a_label_vector_test))


# # Remove NaN and INF in training and eval data

# In[8]:


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


# # Expand the label vectors (one hot encoding)

# In[9]:


#train_labels = to_categorical(train_labels)
#eval_labels = to_categorical(eval_labels)


# # Optimizers

# In[10]:

rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
adam = optimizers.Adam(lr=3e-5, beta_1=0.9, beta_2=0.999,
                       epsilon=None, decay=5e-6, amsgrad=True)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
adamax = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)


# In[11]:

featureSet = train_data
Label = label_training

featureSet = np.split(featureSet, np.array(
    [NumofFeaturetoUse]), axis=2)[0]

print('training data: ' + str(featureSet.shape))
print('training label: ' + str(Label.shape))


# In[12]:

featureSet_val = eval_data
Label_val = label_testing

featureSet_val = np.split(featureSet_val, np.array(
    [NumofFeaturetoUse]), axis=2)[0]

print('evaluation data: ' + str(featureSet_val.shape))
print('evaluation label: ' + str(Label_val.shape))



# In[14]:


def record(str_message, log_file):
    str_message = str_message + '\n'
    file = open(log_file, 'a')
    file.write(str_message)
    file.close()

# In[ ]:
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

    model.compile(loss='binary_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    model.summary()

    return model


#title = 'H_A_neurons_' + str(n_neurons) + '_batches_' + str(n_batch) + '_filters_' + str(
#    nbindex) + '_dropout_' + str(dropout) + '_kerSize_' + str(fillength) + '_dense_' + str(dense_layers)

title = 'All_CNN_ChangedLeaky'
filepath = str(num_layers) + "_Layer(s)//Checkpoint_" + title + ".hdf5"
#model = load_model(filepath)

save_to_path = str(num_layers) + '_Layer(s)//'

if not os.path.exists(save_to_path):
    os.mkdir(save_to_path)

X, X_test, Y, Y_test= model_selection.train_test_split(featureSet, Label, test_size=0.25, shuffle=True)

model = create_cnn(title, num_layers, n_neurons, n_batch,
            nbindex, dropout, classes, dense_layers)

filepath = str(num_layers) + "_Layer(s)//Checkpoint_"+ title + ".hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

early_stopping_monitor = EarlyStopping(patience=50)

callbacks_list = [checkpoint, early_stopping_monitor]

model.fit(X, Y, nb_epoch=n_epoch, batch_size=n_batch,  callbacks=callbacks_list, validation_data=(X_test, Y_test), verbose=1)

model.save_weights(str(num_layers) + "_Layer(s)//" + title + ".hdf5")

model.load_weights(filepath)

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


# In[ ]:

print(accuracy_score(y_true, y_pred))
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(tn)
print(fp)
print(fn)
print(tp)
