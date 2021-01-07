import numpy as np
import os

from sklearn.metrics import f1_score
import tensorflow as tf

#model = keras.models.load_model('path/to/location')
# load training and testing ... 

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
        '''
        if 'Calm' in npy:
            continue
        '''
        current = np.load(path+npy)
        X = np.vstack((X, current))
        label = [files.index(npy)]*len(current)       
        y = y + label       
    X = X[1:]
    y = tf.keras.utils.to_categorical(y)    
    print(X.shape)
    print(y.shape)    
    return X, y


from keras.models import Model, load_model, Sequential

model_path = 'D://GitHub//Patient-Caregiver-Relationship//module//five_class_ood//'
model = load_model(model_path + 'model.hdf5')


X_test = np.load('new_X_3rd.npy')
y_test = np.load('new_y_3rd.npy')

y_trues = [np.argmax(y) for y in y_test]
y_preds = [np.argmax(y) for y in model.predict(X_test)]

f1_macro = f1_score(y_trues, y_preds, average='macro')
print(f1_macro)
f1_micro = f1_score(y_trues, y_preds, average='micro')
print(f1_micro)

f1_weighted = f1_score(y_trues, y_preds, average='weighted')
print(f1_weighted)

f1_average = f1_score(y_trues, y_preds, average='samples')
print(f1_average)