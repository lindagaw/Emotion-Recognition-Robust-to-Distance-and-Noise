'''
In this file, we rearrange the training and testing files in the dataset for EMOTION.
'''
from sklearn.model_selection import train_test_split

import numpy as np
import os
import sys

original_train = 'D://Datasets//TRAINING//padded_deamplified_homenoised_reverberated//npy//train//'
original_test = 'D://Datasets//TRAINING//padded_deamplified_homenoised_reverberated//npy//test//'

new_train = 'D://Datasets//TRAINING//padded_deamplified_homenoised_reverberated//train//'
new_test = 'D://Datasets//TRAINING//padded_deamplified_homenoised_reverberated//test//'
new_val = 'D://Datasets//TRAINING//padded_deamplified_homenoised_reverberated//val//'



def rearrange_only_reverb(original_path, dest, train_or_test, emotion):

    try:
        os.mkdir(dest + '//reverb//')
    except Exception as e:
        print(e)

    for emotion_dir in sorted(os.listdir(original_path)):
        X = []

        if not emotion in emotion_dir:
            continue

        for npy in os.listdir((original_path + emotion_dir)):

            if 'WetDry' not in npy:
                continue
            elif 'deamp' not in npy:

                vec = np.load(original_path + emotion_dir + '//' + npy)
                X.append(vec.tolist())
                print(npy)

        X = np.asarray(X)

        if train_or_test == 'train':
            y = ['NA'] * len(X)
            X_train, X_val, _, _ = train_test_split(X, y, test_size=0.25, random_state=42)

            np.save(dest + '//reverb//X_only_reverb_' + emotion_dir + train_or_test + '.npy', X_train)
            np.save(new_val + '//reverb//X_only_reverb_' + emotion_dir + train_or_test + '.npy', X_val)
        else:
            np.save(dest + '//reverb//X_only_reverb_' + emotion_dir + train_or_test + '.npy', X)

        print(X.shape)



def rearrange_only_deamplified(original_path, dest, train_or_test, emotion):

    try:
        os.mkdir(dest + '//deamp//')
    except Exception as e:
        print(e)
        

    for emotion_dir in sorted(os.listdir(original_path)):
        X = []
        #y = []

        if not emotion in emotion_dir:
            continue

        for npy in os.listdir((original_path + emotion_dir)):

            if 'deamp' not in npy:
                continue
            elif 'WetDry' not in npy:
                vec = np.load(original_path + emotion_dir + '//' + npy)
                X.append(vec.tolist())
                #label = sorted(os.listdir(original_path)).index(emotion_dir)
                #y.append( label )

                print(npy)

        #y = np.expand_dims(np.asarray(y), axis=0)
        X = np.asarray(X)

        if train_or_test == 'train':
            y = ['NA'] * len(X)
            X_train, X_val, _, _ = train_test_split(X, y, test_size=0.25, random_state=42)

            np.save(dest + '//deamp//X_deamp_' + emotion_dir + train_or_test + '.npy', X_train)
            np.save(new_val + '//deamp//X_deamp_' + emotion_dir + train_or_test + '.npy', X_val)
        else:
            np.save(dest + '//deamp//X_deamp_' + emotion_dir + train_or_test + '.npy', X)

        print(X.shape)

def rearrange_only_clean(original_path, dest, train_or_test, emotion):

    try:
        os.mkdir(dest + '//clean//')
    except Exception as e:
        print(e)
        

    for emotion_dir in sorted(os.listdir(original_path)):
        X = []
        #y = []

        if not emotion in emotion_dir:
            continue

        for npy in os.listdir((original_path + emotion_dir)):

            if not 'deamp' in npy and not 'WetDry' in npy:
                vec = np.load(original_path + emotion_dir + '//' + npy)
                X.append(vec.tolist())
                print(npy)

        X = np.asarray(X)

        if train_or_test == 'train':
            y = ['NA'] * len(X)
            X_train, X_val, _, _ = train_test_split(X, y, test_size=0.25, random_state=42)

            np.save(dest + '//clean//X_clean_' + emotion_dir + train_or_test + '.npy', X_train)
            np.save(new_val + '//clean//X_clean_' + emotion_dir + train_or_test + '.npy', X_val)
        else:
            np.save(dest + '//clean//X_clean_' + emotion_dir + train_or_test + '.npy', X)

        print(X.shape)



my_list = ['train']

for train_or_test in my_list:
    if train_or_test == 'train':
        original = original_train
        new_path = new_train
    else:
        original = original_test
        new_path = new_test

    rearrange_only_reverb(original, new_path, train_or_test, sys.argv[1])
    rearrange_only_deamplified(original, new_path, train_or_test, sys.argv[1])
    rearrange_only_clean(original, new_path, train_or_test, sys.argv[1])
