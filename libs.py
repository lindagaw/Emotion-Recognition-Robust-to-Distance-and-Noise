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
