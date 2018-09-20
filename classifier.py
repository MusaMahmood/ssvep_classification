import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Conv1D, Conv2D
from keras.optimizers import Adam
# from keras.utils.vis_utils import plot_model

# y generally needs to be one hot, but that is done in preprocessing (to_categorical)
from sklearn.model_selection import train_test_split

data_directory = 'time_domain_hpf_new/w'
# Options
win_len = 256
# Hyperparams
train_ratio = 0.75


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

    x_train, x_test, y_train, y_test = train_test_split(x_train_data, y_train_data, train_size=train_ratio)
    return x_train, y_train, x_test, y_test


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


#
model = build_model()
"""
with open('modelNote.txt', 'a') as file:
    modelNote = model.to_yaml()
    file.write('\n\n')
    file.write(modelNote)
"""
print(model.summary())
model.fit(x_train, y_train, epochs=100, batch_size=256)
score = model.evaluate(x_test, y_test, batch_size=16)
# model.save("my_model.h")
# plot_model(model, to_file='model.png')
print(score)
