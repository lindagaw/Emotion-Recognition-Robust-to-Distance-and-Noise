import numpy as np
import os
import sys

import h5py
from keras.utils import np_utils, to_categorical
from sklearn.metrics import auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras
import tensorflow as tf

#model = keras.models.load_model('path/to/location')
# load training and testing ... 

dest_all = 'test//all//'
dest_clean = 'test//clean//'
dest_reverb = 'test//reverb//'
dest_deamp = 'test//deamp//'

def load_vectors(path):
    files = sorted(os.listdir(path))
    X = np.expand_dims(np.zeros((48, 272)), axis=0)
    y = []
    for npy in files:
        if 'Calm' in npy:
            continue
        current = np.load(path+npy)
        X = np.vstack((X, current))
        label = [files.index(npy)]*len(current)       
        y = y + label       
    X = X[1:]
    y = to_categorical(y)    
    print(X.shape)
    print(y.shape)    
    return X, y


def mil_squared_error(y_true, y_pred):
    return tf.keras.backend.square(tf.keras.backend.max(y_pred) - tf.keras.backend.max(y_true))

dependencies = {
    'mil_squared_error': mil_squared_error
}

model = keras.models.load_model('D://GitHub//module//five_class_ood//model.hdf5', custom_objects=dependencies)



def eval(model, model_description):
    print(model_description)
    y_preds = [np.argmax(val) for val in model.predict(X_test)]
    y_trues = [np.argmax(val) for val in y_test]

    acc = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds, average='micro')

    print(acc)
    print(f1)

for dest in [dest_clean, dest_reverb, dest_deamp, dest_all]:
    print('dest')
    X_test, y_test = load_vectors(dest)
    eval(model, 'overall f1 score')