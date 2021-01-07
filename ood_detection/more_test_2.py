import os
import shutil
import sys
import h5py

import librosa

import numpy as np
from numpy import array

from sklearn.metrics import auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import tensorflow as tf

def mil_squared_error(y_true, y_pred):
    return tf.keras.backend.square(tf.keras.backend.max(y_pred) - tf.keras.backend.max(y_true))

adam = tf.keras.optimizers.Adam(learning_rate=1e-5)

# load training and testing ... 

condition = 'all'

new_train = '..//train//' + condition + '//'
new_test = '..//test//' + condition + '//'
new_val = '..//val//' + condition + '//'

def load_vectors(path):
    files = sorted(os.listdir(path))
    X = np.expand_dims(np.zeros((48, 272)), axis=0)
    y = []
    for npy in files:
        current = np.load(path+npy)
        X = np.vstack((X, current))
        label = [files.index(npy)]*len(current)       
        y = y + label       
    X = X[1:]
    y = tf.keras.utils.to_categorical(y)    
    print(X.shape)
    print(y.shape)    
    return X, y
    
#X_train, y_train = load_vectors(new_train)
X_test, y_test = load_vectors(new_test)

from tensorflow import keras
model = keras.models.load_model('D://GitHub//module//five_class_ood//model.hdf5', custom_objects={'mil_squared_error': mil_squared_error})

y_preds = model.predict(X_test)
y_preds = [np.argmax(y) for y in y_preds]
y_trues = [np.argmax(y) for y in y_test]

print(f1_score(y_trues, y_preds, average='micro'))
print(f1_score(y_trues, y_preds, average='macro'))
print(f1_score(y_trues, y_preds, average='weighted'))