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

dest = 'voxceleb//'

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
    y = to_categorical(y)    
    print(X.shape)
    print(y.shape)    
    return X, y
    
X_test, y_test = load_vectors(dest)

model_path = 'models//'

models = {}

def mil_squared_error(y_true, y_pred):
    return tf.keras.backend.square(tf.keras.backend.max(y_pred) - tf.keras.backend.max(y_true))

dependencies = {
    'mil_squared_error': mil_squared_error
}


for model_name in os.listdir(model_path):

    if 'deamp' in model_name:
        models['deamp'] = keras.models.load_model(model_path + model_name, custom_objects=dependencies)
    elif 'reverb' in model_name:
        models['reverb'] = keras.models.load_model(model_path + model_name, custom_objects=dependencies)
    elif 'clean' in model_name:
        models['clean'] = keras.models.load_model(model_path + model_name, custom_objects=dependencies)

def eval(model, model_description):
    print(model_description)
    y_preds = [np.argmax(val) for val in model.predict(X_test)]
    y_trues = [np.argmax(val) for val in y_test]

    acc = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds, average='weighted')

    print(acc)
    print(f1)

eval(models['clean'], 'clean model')

eval(models['deamp'], 'deamp model')

eval(models['reverb'], 'reverb model')