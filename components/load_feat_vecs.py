import numpy as np
from keras.utils import to_categorical

h_feature_vector = np.load('Features//h_feature_vector_48.npy')
h_label_vector = np.load('Features//h_label_vector_48.npy')

a_feature_vector = np.load('Features//a_feature_vector_48.npy')
a_label_vector = np.load('Features//a_label_vector_48.npy')

n_feature_vector = np.load('Features//n_feature_vector_48.npy')
n_label_vector = np.load('Features//n_label_vector_48.npy')

s_feature_vector = np.load('Features//s_feature_vector_48.npy')
s_label_vector = np.load('Features//s_label_vector_48.npy')

h_feature_vector_test = np.load('Features//h_feature_vector_test_48.npy')
h_label_vector_test = np.load('Features//h_label_vector_test_48.npy')

a_feature_vector_test = np.load('Features//a_feature_vector_test_48.npy')
a_label_vector_test = np.load('Features//a_label_vector_test_48.npy')

n_feature_vector_test = np.load('Features//n_feature_vector_test_48.npy')
n_label_vector_test = np.load('Features//n_label_vector_test_48.npy')

s_feature_vector_test = np.load('Features//s_feature_vector_test_48.npy')
s_label_vector_test = np.load('Features//s_label_vector_test_48.npy')

h_label_vector[h_label_vector == 0] = 0
a_label_vector[a_label_vector == 1] = 1

h_label_vector_test[h_label_vector_test == 0] = 0
a_label_vector_test[a_label_vector_test == 1] = 1

h_label_vector = to_categorical(h_label_vector, num_classes=2)
a_label_vector = to_categorical(a_label_vector, num_classes=2)

h_label_vector_test = to_categorical(h_label_vector_test, num_classes=2)
a_label_vector_test = to_categorical(a_label_vector_test, num_classes=2)

# Load training npy files
featureSet_training = np.vstack((h_feature_vector, a_feature_vector))
label_training = np.vstack((h_label_vector, a_label_vector))

# Load testing npy files
featureSet_testing = np.vstack((h_feature_vector_test, a_feature_vector_test))
label_testing = np.vstack((h_label_vector_test, a_label_vector_test))

def float_compatible(input_np):

    x = np.where(input_np >= np.finfo(np.float32).max)
    for index in range(0, len(x[0])):
        x_position = x[0][index]
        y_position = x[1][index]

        input_np[x_position, y_position] = 0.0
    input_np = np.nan_to_num(input_np)

    return input_np


train_data = float_compatible((featureSet_training).astype(np.float32))
eval_data = float_compatible((featureSet_testing).astype(np.float32))
