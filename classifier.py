import glob
import os

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Flatten, Conv1D
from keras.models import Sequential
from keras.models import load_model
from scipy.io import loadmat
from scipy.io import savemat
# y generally needs to be one hot, but that is done in preprocessing (to_categorical)
from sklearn.model_selection import train_test_split

import tf_shared_k as tfs

data_directory = 'time_domain_hpf_new/w'
# Options
win_len = 512
# Hyperparams
train_ratio = 0.75
DATASET = 'ssvep_' + str(win_len)

description = DATASET + '_annotate'
keras_model_name = description + '.h5'
model_dir = tfs.prep_dir('model_exports/')
keras_file_location = model_dir + keras_model_name
# Start Timer:
start_time_ms = tfs.current_time_ms()

# Setup:
TRAIN = False
TEST = True
SAVE_PREDICTIONS = False
SAVE_HIDDEN = True
# EXPORT_OPT_BINARY = False


def load_data(data_directory, image_shape, key_x, key_y):
    x_train_data = np.empty([0, *image_shape], np.float32)
    y_train_data = np.empty([0], np.float32)
    training_files = glob.glob(data_directory + "/*.mat")
    for f in training_files:
        x_array = loadmat(f).get(key_x)
        y_array = loadmat(f).get(key_y)
        y_array = y_array.reshape([np.amax(y_array.shape)])
        x_train_data = np.concatenate((x_train_data, x_array), axis=0)
        y_train_data = np.concatenate((y_train_data, y_array), axis=0)
    y_train_data = np.asarray(pd.get_dummies(y_train_data).values).astype(np.float32)

    print("Loaded Data Shape: X:", x_train_data.shape, " Y: ", y_train_data.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_train_data, y_train_data, train_size=train_ratio,
                                                        random_state=1)
    return x_train, y_train, x_test, y_test


#
batch_size = 256
epochs = 60
output_folder = 'data_out/' + description + '/'

# To load the data:
input_shape = [2, win_len]

x_train, y_train, x_test, y_test = load_data(data_directory + str(win_len), input_shape, key_x='relevant_data',
                                             key_y='Y')

# Tranposing for conv
x_train = np.transpose(x_train, (0, 2, 1))
x_test = np.transpose(x_test, (0, 2, 1))
input_shape = [win_len, 2]
print(input_shape)

print("Processed Data Shape: X:", x_train.shape, " Y: ", y_train.shape)
print("Processed Test Shape: X:", x_test.shape, " Y: ", y_test.shape)


def build_model():
    model = Sequential()
    model.add(Conv1D(64, 8, strides=2, activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv1D(128, 8, strides=2, activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(output_dim=5, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
    return model


model = []
# tf_backend.set_session(tfs.get_session(0.75))
# with tf.device('/gpu:0'):
if TRAIN:
    if os.path.isfile(keras_file_location):
        model = load_model(keras_file_location)
    else:
        model = build_model()
    print(model.summary())

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    model.save(keras_file_location)

if os.path.isfile(keras_file_location):
    if not TRAIN:
        model = load_model(keras_file_location)
        print(model.summary())
        if TEST:
            score, acc = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
            print('Test score: {} , Test accuracy: {}'.format(score, acc))
            y_prob = model.predict(x_test)
            tfs.print_confusion_matrix_v2(y_prob, y_test)
    else:
        if TEST and model is not None:
            score, acc = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
            print('Test score: {} , Test accuracy: {}'.format(score, acc))
            y_prob = model.predict(x_test)
            tfs.print_confusion_matrix_v2(y_prob, y_test)
        else:
            print('This should never happen: model does not exist')
            exit(-1)
else:
    print("Model Not Found!")
    if not TRAIN:
        exit(-1)

if SAVE_PREDICTIONS:
    # predict
    yy_probabilities = model.predict(x_test, batch_size=batch_size)
    yy_predicted = tfs.maximize_output_probabilities_v2(yy_probabilities)
    data_dict = {'x_val': x_test, 'y_val': y_test, 'y_prob': yy_probabilities, 'y_out': yy_predicted}
    savemat(tfs.prep_dir(output_folder) + description + '.mat', mdict=data_dict)

if SAVE_HIDDEN:
    layers_of_interest = ['conv1d_1', 'conv1d_2', 'flatten_1', 'dense_1', 'dense_2']
    # np.random.seed(0)
    # rand_indices = np.random.randint(0, x_test.shape[0], 250)
    print('Saving hidden layers: ', layers_of_interest)
    tfs.get_keras_layers(model, layers_of_interest, x_test, y_test,
                         output_dir=tfs.prep_dir(output_folder + '/hidden/'),
                         fname='h_' + description + '.mat')

# TODO: Save hidden Layers
print('Elapsed Time (ms): ', tfs.current_time_ms() - start_time_ms)
print('Elapsed Time (min): ', (tfs.current_time_ms() - start_time_ms) / 60000)
