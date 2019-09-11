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

sample_rate=44100
hop_length = 441  # frame size= 2*hop
segment_length=int(sample_rate*0.2)  #0.2
segment_pad=int(sample_rate*0.02)     #0.02
overlappiong=int(sample_rate*0.1)   #0.1

NumofFeaturetoUse = 100 # this will re-assigned for different classifiers
frame_number = 48

# input new indices file here
indices_filename = 'D://indices_filename.npy'
indices=np.load(indices_filename)

h_TESS = 'D://Datasets//TESS//PAD_REVERB_NOISE//Happy//'
a_TESS = 'D://Datasets//TESS//PAD_REVERB_NOISE//Angry//'
n_TESS = 'D://Datasets//TESS//PAD_REVERB_NOISE//Neutral//'
s_TESS = 'D://Datasets//TESS//PAD_REVERB_NOISE//Sad//'

h_EMO = 'D://Datasets//EMO-DB//wav//Happy//'
a_EMO = 'D://Datasets//EMO-DB//wav//Angry//'
n_EMO = 'D://Datasets//EMO-DB//wav//Neutral//'
s_EMO = 'D://Datasets//EMO-DB//wav//Sad//'
o_EMO = 'D://Datasets//EMO-DB//wav//Other//'

test_Happy = 'D://Datasets//TRAINING//Happy_test//'
test_Angry = 'D://Datasets//TRAINING//Angry_test//'
test_Neutral = 'D://Datasets//TRAINING//Neutral_test//'
test_Sad = 'D://Datasets//TRAINING//Sad_test//'

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

def function_FeatureExtractfromSinglewindow(y,hop_length,sr):

    genFeatures=np.array([])

    mfcc0 = librosa.feature.mfcc(y=y, sr=sr, n_fft=hop_length*2, hop_length=hop_length, n_mfcc=13)
    mfcc=np.transpose(mfcc0)

    genFeatures = np.hstack((genFeatures, np.amin(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, np.median(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, np.std(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, np.var(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(mfcc, 0)))
    #print(genFeatures.shape)

    mfcc_delta=librosa.feature.delta(mfcc0)
    mfcc_delta=np.transpose(mfcc_delta)
    genFeatures = np.hstack((genFeatures, np.amin(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.median(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.std(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.var(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(mfcc_delta, 0)))
    #print(genFeatures.shape)

    zcr0=librosa.feature.zero_crossing_rate(y=y, frame_length=hop_length*2, hop_length=hop_length)
    zcr=np.transpose(zcr0)
    genFeatures = np.hstack((genFeatures, np.amin(zcr, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(zcr, 0)))
    genFeatures = np.hstack((genFeatures, np.median(zcr, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(zcr, 0)))
    genFeatures = np.hstack((genFeatures, np.std(zcr, 0)))
    genFeatures = np.hstack((genFeatures, np.var(zcr, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(zcr, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(zcr, 0)))
    #print(genFeatures.shape)

    zcr_delta=librosa.feature.delta(zcr0)
    zcr_delta=np.transpose(zcr_delta)
    genFeatures = np.hstack((genFeatures, np.amin(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.median(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.std(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.var(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(zcr_delta, 0)))
    #print(genFeatures.shape)

    Erms0=librosa.feature.rms(y=y, frame_length=hop_length*2, hop_length=hop_length)
    Erms=np.transpose(Erms0)
    genFeatures = np.hstack((genFeatures, np.amin(Erms, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(Erms, 0)))
    genFeatures = np.hstack((genFeatures, np.median(Erms, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(Erms, 0)))
    genFeatures = np.hstack((genFeatures, np.std(Erms, 0)))
    genFeatures = np.hstack((genFeatures, np.var(Erms, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(Erms, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(Erms, 0)))
    #print(genFeatures.shape)

    Erms_delta=librosa.feature.delta(Erms0)
    Erms_delta=np.transpose(Erms_delta)
    genFeatures = np.hstack((genFeatures, np.amin(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.median(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.std(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.var(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(Erms_delta, 0)))
    #print(genFeatures.shape)

    cent0 = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=hop_length*2, hop_length=hop_length)
    cent=np.transpose(cent0)
    genFeatures = np.hstack((genFeatures, np.amin(cent, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(cent, 0)))
    genFeatures = np.hstack((genFeatures, np.median(cent, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(cent, 0)))
    genFeatures = np.hstack((genFeatures, np.std(cent, 0)))
    genFeatures = np.hstack((genFeatures, np.var(cent, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(cent, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(cent, 0)))
    #print(genFeatures.shape)

    cent_delta=librosa.feature.delta(cent0)
    cent_delta=np.transpose(cent_delta)
    genFeatures = np.hstack((genFeatures, np.amin(cent_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(cent_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.median(cent_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(cent_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.std(cent_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.var(cent_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(cent_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(cent_delta, 0)))
    #print(genFeatures.shape)
    #Each frame of a magnitude spectrogram is normalized and treated as a distribution over frequency bins, from which the mean (centroid) is extracted per frame.

    ############### pitch at certain frame
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=8000, n_fft=hop_length*2, hop_length=hop_length)
    p=[pitches[magnitudes[:,i].argmax(),i] for i in range(0,pitches.shape[1])]
    pitch0=np.array(p)   #shape (305,)
    pitch=np.transpose(pitch0)
    genFeatures = np.hstack((genFeatures, np.amin(pitch, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(pitch, 0)))
    genFeatures = np.hstack((genFeatures, np.median(pitch, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(pitch, 0)))
    genFeatures = np.hstack((genFeatures, np.std(pitch, 0)))
    genFeatures = np.hstack((genFeatures, np.var(pitch, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(pitch, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(pitch, 0)))
    #print(genFeatures.shape)

    pitch_delta=librosa.feature.delta(pitch0)
    pitch_delta=np.transpose(pitch_delta)
    genFeatures = np.hstack((genFeatures, np.amin(pitch_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(pitch_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.median(pitch_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(pitch_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.std(pitch_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.var(pitch_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(pitch_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(pitch_delta, 0)))
    #print(genFeatures.shape)    #272
    return genFeatures


'''
Extract specified amount of features from an audio file
'''
def function_FeatureExtract1(audiofile, NumofFeatures):
    extension = '.wav'
    flag_start_all = 0
    flag_Y_start = 0
    All = np.array([])
    NumofFeaturetoUse = NumofFeatures #needs to be reassigned, takes two parameters
    ListOfFrame2Vec = np.empty((0, frame_number, NumofFeaturetoUse))
    audio, s_rate = librosa.load(audiofile, sr=sample_rate)
    segment_start_flag = 0
    start_seg = 0
    while (start_seg + segment_length) < len(audio):
        flag = 1
        sound1 = audio[start_seg:(start_seg + segment_length)]

        featureSet = function_FeatureExtractfromSinglewindow(sound1, hop_length, sample_rate)

        if segment_start_flag == 0:
            SegAllFeat = featureSet
            segment_start_flag = 1
        else:
            SegAllFeat = np.vstack((SegAllFeat, featureSet))

        start_seg = start_seg + overlappiong

    if segment_start_flag == 1:
        #print(SegAllFeat.shape)
        SegAllFeat = normalize(SegAllFeat, norm='l2', axis=0)

    #print(SegAllFeat.shape)
    if flag_start_all == 0:
        All = SegAllFeat
        flag_start_all = 1
    else:
        All = np.vstack((All, SegAllFeat))

    return All

'''
Extract specified amount of features from an audio file
'''
def function_FeatureExtract1(audiofile, NumofFeatures):
    extension = '.wav'
    flag_start_all = 0
    flag_Y_start = 0
    All = np.array([])
    NumofFeaturetoUse = NumofFeatures #needs to be reassigned, takes two parameters
    ListOfFrame2Vec = np.empty((0, frame_number, NumofFeaturetoUse))
    audio, s_rate = librosa.load(audiofile, sr=sample_rate)
    segment_start_flag = 0
    start_seg = 0
    while (start_seg + segment_length) < len(audio):
        flag = 1
        sound1 = audio[start_seg:(start_seg + segment_length)]

        featureSet = function_FeatureExtractfromSinglewindow(sound1, hop_length, sample_rate)

        if segment_start_flag == 0:
            SegAllFeat = featureSet
            segment_start_flag = 1
        else:
            SegAllFeat = np.vstack((SegAllFeat, featureSet))

        start_seg = start_seg + overlappiong

    if segment_start_flag == 1:
        #print(SegAllFeat.shape)
        SegAllFeat = normalize(SegAllFeat, norm='l2', axis=0)

    #print(SegAllFeat.shape)
    if flag_start_all == 0:
        All = SegAllFeat
        flag_start_all = 1
    else:
        All = np.vstack((All, SegAllFeat))

    return All

######################################################################

def h_a_function_FeatureExtract3(InputFolderName):
    X = function_FeatureExtract1(InputFolderName, NumofFeaturetoUse)
    y_pred = model.predict(X)
    print(y_pred)
    x = maxima(list(y_pred[0]))
    return x

def h_a_classifier_eval(emotionFolders):
    print('Predicted with overall classifier: ' + final_filepath)

    correct = 0
    incorrect = 0
    for emotionFolder in emotionFolders:
        if 'Happy' in emotionFolder: val = 0
        elif 'Angry' in emotionFolder: val = 1

        for emotionfile in os.listdir(emotionFolder):
            cond1 = 'deamp_' not in emotionfile and 'WetDry_' not in emotionfile
            cond2 = 'noise' not in emotionfile
            cond4 = emotionfile[0] != '.'
            if cond2 and cond4 and cond1:
                # print(correct+incorrect)
                x = h_a_function_FeatureExtract3(InputFolderName=emotionFolder+emotionfile)
                if(x == val): correct += 1
                else: incorrect += 1
        
    return correct/(correct+incorrect)


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

prefix = 'C://Users//yg9ca//Documents//'
final_filepath = prefix + 'modules//Checkpoint_H_A_neurons_4096_filters_256_dropout_0.2_epoch_50000.hdf5'

model = load_model(final_filepath)
h_a_classifier_eval([test_Happy])
