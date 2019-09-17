import os
import random
from pydub import AudioSegment
from xml.dom import minidom
import numpy as np
import pandas as pd
from pysndfx import AudioEffectsChain
from librosa import load
import shutil
import sys

sys.path.insert(1, '..//components//')
import load_feat_directories

def delete_directory(path):
    shutil.rmtree(path, ignore_errors=True)

def partition_directory(path, new_path, percent):

    if not os.path.exists(new_path):
        os.mkdir(new_path)
    else:
        delete_directory(new_path)
        os.makdir(new_path)

    total = len(os.listdir(path))
    test_number = int(total * percent)
    test_set = np.random.permutation(total)[0:test_number]

    for index in test_set:
        try:
            if os.listdir(path)[index].endswith('.wav'):
                original_location = path + os.listdir(path)[index]
                new_location = new_path + os.listdir(path)[index]
                os.rename(original_location, new_location)
                print(new_location)
        except:
            pass

    print('# of training data + # of testing data = ' + str(total))
    print(str(1 - percent) + ' of the data is in the training set')
    print(str(percent) + ' of the data is in the training testing set')
    print('# of testing data = ' + str(test_set.shape))

# allnoised_npy[0, 1, 2, 3, 4] ==> H, A, N, S, O
# homenoised_npy[0, 1, 2, 3, 4] ==> H, A, N, S, O

all_noised_npy = load_feat_directories.allnoised_npy
allnoised_npy_test = load_feat_directories.allnoised_npy_test

home_noised_npy = load_feat_directories.homenoised_npy
home_noised_npy_test = load_feat_directories.homenoised_npy_test

for index in range(0, 5):
    #x = os.path.exists(all_noised_npy[index])
    #y = os.path.exists(home_noised_npy[index])

    if not os.path.exists(all_noised_npy[index]):
        print(all_noised_npy[index] + ' does not exist. Breaking the loop... ')
    
    if not os.path.exists(home_noised_npy[index]):
        print(home_noised_npy[index] + 'does not exist. Breaking the loop... ')
    

for index in range(0, 5):
    percent = 0.2
    partition_directory(all_noised_npy[index], allnoised_npy_test[index], percent)

for index in range(0, 5):
    percent = 0.2
    partition_directory(home_noised_npy[index], home_noised_npy_test[index], percent)

'''
percent = 0.2
partition_directory(h_directory, h_test, percent)
partition_directory(a_directory, a_test, percent)
partition_directory(n_directory, n_test, percent)
partition_directory(s_directory, s_test, percent)
partition_directory(o_directory, o_test, percent)
'''
