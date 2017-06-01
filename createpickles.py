# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Default module to train a xor classifier and write weights to disk."""

from keras.models import Sequential
from keras.layers.core import Dense,  Activation
import keras.optimizers as kop
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
try:
    import cPickle as pickle
except Exception as ex:
    import pickle


def check_dir_exists(dirname='./pickles'):
    """Check if given dirname exists This will contain all the pickle files."""
    if not os.path.exists(dirname):
        print("Directory to store pickes does not exist. Creating one now: ./pickles")
        os.mkdir(dirname)


def save_x_y_scalar(X_train, Y_train):
    """Use a normalization method on your current dataset and save the coefficients.

    Args:
        X_train:    Input X_train
        Y_train:    Lables Y_train
    Returns:
        Normalized X_train,Y_train ( currently using StandardScaler from scikit-learn)
    """
    scalar_x = StandardScaler()
    X_train = scalar_x.fit_transform(X_train)

    scalar_y = StandardScaler()
    Y_train = scalar_y.fit_transform(Y_train)

    print('dumping StandardScaler objects ..')
    pickle.dump(scalar_y,
                open('pickles/scalar_y.pickle', "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(scalar_x,
                open('pickles/scalar_x.pickle', "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
    return X_train, Y_train


def create_model(X_train, Y_train):
    """create_model will create a very simple neural net model and  save the weights in a predefined directory.

    Args:
        X_train:    Input X_train
        Y_train:    Lables Y_train
    """
    xin = X_train.shape[1]

    model = Sequential()
    model.add(Dense(output_dim=4, input_shape=(xin, )))
    model.add(Activation('tanh'))
    model.add(Dense(4))
    model.add(Activation('linear'))
    model.add(Dense(1))

    rms = kop.RMSprop()

    print('compiling now..')
    model.compile(loss='mse', optimizer=rms)

    model.fit(X_train, Y_train, nb_epoch=1000, batch_size=1, verbose=2)
    score = model.evaluate(X_train, Y_train, batch_size=1)
    print("Evaluation results:", score)
    open('pickles/my_model_architecture.json', 'w').write(model.to_json())

    print("Saving weights in: ./pickles/my_model_weights.h5")
    model.save_weights('pickles/my_model_weights.h5')


if __name__ == '__main__':
    X_train = np.array([[1., 1.], [1., 0], [0, 1.], [0, 0]])
    Y_train = np.array([[0.], [1.], [1.], [0.]])

    check_dir_exists(dirname='./pickles')
    X_train, Y_train = save_x_y_scalar(X_train, Y_train)
    create_model(X_train, Y_train)
