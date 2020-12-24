'''
In this file, we rearrange the training and testing files in the dataset for EMOTION.
'''
from sklearn.model_selection import train_test_split

import numpy as np
import os
import sys

voxceleb_test_raw = 'D://Datasets//VoxCeleb//usable//npy//test//'

dest = 'voxceleb//'

def rearrange(original_path, dest, emotion):

    try:
        os.mkdir(dest)
    except Exception as e:
        print(e)
    count = 0
    for emotion_dir in sorted(os.listdir(original_path)):
        X = []
        #y = []
        if not emotion in emotion_dir:
            continue
        for npy in os.listdir((original_path + emotion_dir)):            
            vec = np.load(original_path + emotion_dir + '//' + npy)
            X.append(vec.tolist())
            print(npy)

            count += 1

        X = np.asarray(X)
        np.save(dest + 'voxceleb_' + emotion + '.npy', X)

        print(X.shape)

rearrange(voxceleb_test_raw, dest, sys.argv[1])