import os
import random
from pydub import AudioSegment
from xml.dom import minidom
import numpy as np
import pandas as pd
from pysndfx import AudioEffectsChain
from librosa import load
import shutil


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
    print(str(percent) + ' of the data is in training set')
    print('# of training data = ' + str(test_set.shape))


'''
percent = 0.2
partition_directory(h_directory, h_test, percent)
partition_directory(a_directory, a_test, percent)
partition_directory(n_directory, n_test, percent)
partition_directory(s_directory, s_test, percent)
partition_directory(o_directory, o_test, percent)
'''
