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

from IPython.display import clear_output

import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sample_rate = 44100
hop_length = 441  # frame size= 2 * hop
segment_length = int(sample_rate * 0.2)  # 0.2
segment_pad = int(sample_rate * 0.02)  # 0.02
overlapping = int(sample_rate * 0.1)  # 0.1

NumofFeaturetoUse = 272
frame_number = 48

try:
    NumofFeaturetoUse = int(sys.argv[1])
    print('Number of features to use is set to ' + str(sys.argv[1]))
except:
    print('Number of features are unspecified. Defaut is set to = 272.')


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
    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format(
        "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


def str_to_int(input_np):
    output_np = []
    for x in np.nditer(input_np):
        if x == 'H':
            x = 0
        elif x == 'A':
            x = 1
        elif x == 'N':
            x = 2
        else:
            x = 3
        output_np.append(x)
    output_np = np.array(output_np)
    output_np = np.reshape(output_np, (len(output_np), 1))
    return output_np


def float_compatible(input_np):
    input_np = np.nan_to_num(input_np)
    x = np.where(input_np >= np.finfo(np.float32).max)
    for index in range(0, len(x[0])):
        try:
            x_position = x[0][index]
            y_position = x[1][index]
            input_np[x_position, y_position] = 0.0
        except:
            print(x)
            print(x[0])
    return input_np


def function_FeatureExtractfromSinglewindow(y, hop_length, sr):
    genFeatures = np.array([])

    mfcc0 = librosa.feature.mfcc(
        y=y, sr=sr, n_fft=hop_length*2, hop_length=hop_length, n_mfcc=13)
    mfcc = np.transpose(mfcc0)
    genFeatures = np.hstack((genFeatures, np.amin(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, np.median(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, np.std(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, np.var(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(mfcc, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(mfcc, 0)))
    #print(genFeatures.shape)

    mfcc_delta = librosa.feature.delta(mfcc0)
    mfcc_delta = np.transpose(mfcc_delta)
    genFeatures = np.hstack((genFeatures, np.amin(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.median(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.std(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.var(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(mfcc_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(mfcc_delta, 0)))
    #print(genFeatures.shape)

    zcr0 = librosa.feature.zero_crossing_rate(
        y=y, frame_length=hop_length*2, hop_length=hop_length)
    zcr = np.transpose(zcr0)
    genFeatures = np.hstack((genFeatures, np.amin(zcr, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(zcr, 0)))
    genFeatures = np.hstack((genFeatures, np.median(zcr, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(zcr, 0)))
    genFeatures = np.hstack((genFeatures, np.std(zcr, 0)))
    genFeatures = np.hstack((genFeatures, np.var(zcr, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(zcr, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(zcr, 0)))
    #print(genFeatures.shape)

    zcr_delta = librosa.feature.delta(zcr0)
    zcr_delta = np.transpose(zcr_delta)
    genFeatures = np.hstack((genFeatures, np.amin(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.median(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.std(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.var(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(zcr_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(zcr_delta, 0)))
    #print(genFeatures.shape)

    Erms0 = librosa.feature.rms(
        y=y, frame_length=hop_length*2, hop_length=hop_length)
    Erms = np.transpose(Erms0)
    genFeatures = np.hstack((genFeatures, np.amin(Erms, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(Erms, 0)))
    genFeatures = np.hstack((genFeatures, np.median(Erms, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(Erms, 0)))
    genFeatures = np.hstack((genFeatures, np.std(Erms, 0)))
    genFeatures = np.hstack((genFeatures, np.var(Erms, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(Erms, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(Erms, 0)))
    #print(genFeatures.shape)

    Erms_delta = librosa.feature.delta(Erms0)
    Erms_delta = np.transpose(Erms_delta)
    genFeatures = np.hstack((genFeatures, np.amin(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.median(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.std(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, np.var(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(Erms_delta, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(Erms_delta, 0)))
    #print(genFeatures.shape)

    cent0 = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=hop_length*2, hop_length=hop_length)
    cent = np.transpose(cent0)
    genFeatures = np.hstack((genFeatures, np.amin(cent, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(cent, 0)))
    genFeatures = np.hstack((genFeatures, np.median(cent, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(cent, 0)))
    genFeatures = np.hstack((genFeatures, np.std(cent, 0)))
    genFeatures = np.hstack((genFeatures, np.var(cent, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(cent, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(cent, 0)))
    #print(genFeatures.shape)

    cent_delta = librosa.feature.delta(cent0)
    cent_delta = np.transpose(cent_delta)
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
    pitches, magnitudes = librosa.core.piptrack(
        y=y, sr=sr, fmin=75, fmax=8000, n_fft=hop_length*2, hop_length=hop_length)
    p = [pitches[magnitudes[:, i].argmax(), i]for i in range(0, pitches.shape[1])]
    pitch0 = np.array(p)  # shape (305,)
    pitch = np.transpose(pitch0)
    genFeatures = np.hstack((genFeatures, np.amin(pitch, 0)))
    genFeatures = np.hstack((genFeatures, np.amax(pitch, 0)))
    genFeatures = np.hstack((genFeatures, np.median(pitch, 0)))
    genFeatures = np.hstack((genFeatures, np.mean(pitch, 0)))
    genFeatures = np.hstack((genFeatures, np.std(pitch, 0)))
    genFeatures = np.hstack((genFeatures, np.var(pitch, 0)))
    genFeatures = np.hstack((genFeatures, st.skew(pitch, 0)))
    genFeatures = np.hstack((genFeatures, st.kurtosis(pitch, 0)))
    #print(genFeatures.shape)

    pitch_delta = librosa.feature.delta(pitch0)
    pitch_delta = np.transpose(pitch_delta)
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


# Extract specified amount of features from an audio file
def extract_feats_single_wav(npy_path, audiofile):
    flag_start_all = 0
    All = np.array([])
    audio, s_rate = librosa.load(audiofile, sr=sample_rate)
    segment_start_flag = 0
    start_seg = 0
    while(start_seg + segment_length) < len(audio):
        sound_window = audio[start_seg:(start_seg + segment_length)]
        featureSet = function_FeatureExtractfromSinglewindow(
            sound_window, hop_length, s_rate)
        if segment_start_flag == 0:
            SegAllFeat = featureSet
            segment_start_flag = 1
        else:
            SegAllFeat = np.vstack((SegAllFeat, featureSet))
        start_seg = start_seg + overlapping
    
    SegAllFeat = float_compatible(SegAllFeat)
    SegAllFeat = normalize(SegAllFeat, norm='l2', axis=0)

    if flag_start_all == 0:
        All = SegAllFeat
        flag_start_all = 1
    else:
        All = np.vstack((All, SegAllFeat))
    
    audio_npy = (audiofile[:len(audiofile)-4] + '.npy').split('//')[len((audiofile[:len(audiofile)-4] + '.npy').split('//'))-1]
    audio_npy = npy_path + audio_npy
    All = float_compatible(All)
    '''
    if str(All.shape) == '(48, 272)':
        np.save(audio_npy, All)
        print('saved')
    '''
    return All


allnoised_happy = '..//..//..//Datasets//padded_deamplified_allnoised_reverberated//Happy//'
allnoised_happy_npy = '..//..//..//Datasets//padded_deamplified_allnoised_reverberated//npy//Happy_npy//'
allnoised_angry = '..//..//..//Datasets//padded_deamplified_allnoised_reverberated//Angry//'
allnoised_angry_npy = '..//..//..//Datasets//padded_deamplified_allnoised_reverberated//npy//Angry_npy//'
allnoised_neutral = '..//..//..//Datasets//padded_deamplified_allnoised_reverberated//Neutral//'
allnoised_neutral_npy = '..//..//..//Datasets//padded_deamplified_allnoised_reverberated//npy//Neutral_npy//'
allnoised_sad = '..//..//..//Datasets//padded_deamplified_allnoised_reverberated//Sad//'
allnoised_sad_npy = '..//..//..//Datasets//padded_deamplified_allnoised_reverberated//npy//Sad_npy//'
allnoised_other = '..//..//..//Datasets//padded_deamplified_allnoised_reverberated//Other//'
allnoised_other_npy = '..//..//..//Datasets//padded_deamplified_allnoised_reverberated//npy//Other_npy//'

allnoised = [allnoised_happy, allnoised_angry, allnoised_neutral, allnoised_sad, allnoised_other]
allnoised_npy = [allnoised_happy_npy, allnoised_angry_npy,
                 allnoised_neutral_npy, allnoised_sad_npy, allnoised_other_npy]

homenoised_happy = '..//..//..//Datasets//padded_deamplified_homenoised_reverberated//Happy//'
homenoised_happy_npy = '..//..//..//Datasets//padded_deamplified_homenoised_reverberated//npy//Happy_npy//'
homenoised_angry = '..//..//..//Datasets//padded_deamplified_homenoised_reverberated//Angry//'
homenoised_angry_npy = '..//..//..//Datasets//padded_deamplified_homenoised_reverberated//npy//Angry_npy//'
homenoised_neutral = '..//..//..//Datasets//padded_deamplified_homenoised_reverberated//Neutral//'
homenoised_neutral_npy = '..//..//..//Datasets//padded_deamplified_homenoised_reverberated//npy//Neutral_npy//'
homenoised_sad = '..//..//..//Datasets//padded_deamplified_homenoised_reverberated//Sad//'
homenoised_sad_npy = '..//..//..//Datasets//padded_deamplified_homenoised_reverberated//npy//Sad_npy//'
homenoised_other = '..//..//..//Datasets//padded_deamplified_homenoised_reverberated//Other//'
homenoised_other_npy = '..//..//..//Datasets//padded_deamplified_homenoised_reverberated//npy//Other_npy//'

homenoised = [homenoised_happy, homenoised_angry, homenoised_neutral, homenoised_sad, homenoised_other]
homenoised_npy = [homenoised_happy_npy, homenoised_angry_npy,
                  homenoised_neutral_npy, homenoised_sad_npy, homenoised_other_npy]

# index 0 - happy, index 1 - angry, index 2 - neutral, index 3 - sad, index 4 - other


emodb_Happy = 'D://Datasets//EMO-DB//wav//Happy//'
emodb_Angry = 'D://Datasets//EMO-DB//wav//Angry//'
emodb_Neutral = 'D://Datasets//EMO-DB//wav//Neutral//'
emodb_Sad = 'D://Datasets//EMO-DB//wav//Sad//'
emodb_Other = 'D://Datasets//EMO-DB//wav//Other//'

emodb_Happy_npy = 'D://Datasets//EMO-DB//wav//npy//Happy//'
emodb_Angry_npy = 'D://Datasets//EMO-DB//wav//npy//Angry//'
emodb_Neutral_npy = 'D://Datasets//EMO-DB//wav//npy//Neutral//'
emodb_Sad_npy = 'D://Datasets//EMO-DB//wav//npy//Sad//'
emodb_Other_npy = 'D://Datasets//EMO-DB//wav//npy//Other//'

emodb = [emodb_Happy, emodb_Angry, emodb_Neutral, emodb_Sad, emodb_Other]
emodb_npy = [emodb_Happy_npy, emodb_Angry_npy, emodb_Neutral_npy, emodb_Sad_npy, emodb_Other_npy]

CaFE_happy = 'D://Datasets//CaFE//Happy//'
CaFE_angry = 'D://Datasets//CaFE//Angry//'
CaFE_neutral = 'D://Datasets//CaFE//Neutral//'
CaFE_sad = 'D://Datasets//CaFE//Sad//'

CaFE_happy_npy = 'D://Datasets//CaFE//npy//Happy//'
CaFE_angry_npy = 'D://Datasets//CaFE//npy//Angry//'
CaFE_neutral_npy = 'D://Datasets//CaFE//npy//Neutral//'
CaFE_sad_npy = 'D://Datasets//CaFE//npy//Sad//'

CaFE = [CaFE_happy, CaFE_angry, CaFE_neutral, CaFE_sad]
CaFE_npy = [CaFE_happy_npy, CaFE_angry_npy, CaFE_neutral_npy, CaFE_sad_npy]

elapsed = []
for index in [0, 1, 2, 3, 4]:
    for audio in os.listdir(homenoised[index]):
        npy_title = homenoised_npy[index] + audio[:len(audio)-4] + '.npy'
        try:
            if os.path.isfile(npy_title):
                print(npy_title + 'already exists. Skipping...')
                continue
            elif not audio.endswith('.wav') or audio[0] == '.':
                continue
            else:
                audio = homenoised[index] + audio
                start = time.time()
                extract_feats_single_wav(homenoised_npy[index], audio)
                end = time.time()
                elapsed.append(end-start)
        except Exception as e:
            print(e)

'''
for index in [0, 1, 2, 3]:
    for audio in os.listdir(emodb[index]):
        npy_title = emodb_npy[index] + audio[:len(audio)-4] + '.npy'
        try:
            if os.path.isfile(npy_title):
                print(npy_title + 'already exists. Skipping...')
                continue
            elif not audio.endswith('.wav') or audio[0] == '.':
                continue
            else:
                audio = emodb[index] + audio
                extract_feats_single_wav(emodb_npy[index], audio)
                print(npy_title + ' created.')
        except Exception as e:
            print(e)
'''
'''
for index in [4]:
    for audio in os.listdir(allnoised[index]):
        print(audio)
        print(allnoised[index])

        npy_title = allnoised_npy[index] + audio[:len(audio)-4] + '.npy'
        print(npy_title)
        if os.path.isfile(npy_title):
            print(npy_title + 'already exists. Skipping...')
            continue
        elif not audio.endswith('.wav') or audio[0] == '.':
            continue
        else:
            audio = allnoised[index] + audio
            extract_feats_single_wav(allnoised_npy[index], audio)
            
for index in [4]:
    for audio in os.listdir(homenoised[index]):
        npy_title = homenoised_npy[index] + audio[:len(audio)-4] + '.npy'
        try:
            if os.path.isfile(npy_title):
                print(npy_title + 'already exists. Skipping...')
                continue
            elif not audio.endswith('.wav') or audio[0] == '.':
                continue
            else:
                audio = homenoised[index] + audio
                extract_feats_single_wav(homenoised_npy[index], audio)
        except:
            pass
'''