# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:16:55 2024

@author: Juanjo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

import models.RNN as RNN
import dataset.pendulum as pendulum
import utils.methods as methods
def cross_validation(k, data, labels):
    """
    

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    x_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    x_val : TYPE
        DESCRIPTION.
    y_val : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    tscv = TimeSeriesSplit(n_splits = k)
    scores = []
    
    
    for train_index, val_index in tscv.split(data):
        # split the data
        x_train, x_val = data[train_index], data[val_index]
        y_train, y_val = labels[train_index], labels[val_index]
        
        # define the model
        model = RNN.RNNmodel(input_shape = (x_train.shape[0], x_train.shape[1]), units = 40, output_size = 4)
        model.compile(optimizer = 'adam', loss = 'mse')
        model.summary()

        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 4))
        y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))

        x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 4))
        y_val = y_val.reshape((y_val.shape[0], 1, y_val.shape[1]))

        history = model.fit(x_train, y_train, shuffle = False, epochs = 3, batch_size = 64, validation_data = (x_val, y_val))
        plt.plot(history.history['loss'], label = 'training loss')
        plt.plot(history.history['val_loss'], label = 'validation loss')
        plt.legend()
        
        scores.append(model.evaluate(x_val, y_val, verbose = 0))
    
    avg_score = np.mean(scores)
    plt.show()
    
    print('Media de las puntuaciones: ', avg_score)
    return scores, avg_score

if __name__ == "__main__":
    # generate the data
    state = [np.random.rand()*np.pi, np.random.rand()*np.pi, np.random.rand()*np.pi, np.random.rand()*np.pi]
    t_stop = 1000.0
    dt = 0.02
    t = np.arange(0, t_stop, dt)
    p = pendulum.Pendulum(t_stop)
    data = p.solver(state)
    
    num_train_samples = int(0.75*len(data))
    num_val_samples = len(data) - num_train_samples

    
    # scale the data
    scaler = MinMaxScaler(feature_range = (0, 1))
    data_sc = scaler.fit_transform(data)
    
    # create sequences
    n_steps = 30
    x_train, y_train = methods.create_sequences(data_sc[:num_train_samples], n_steps)
    x_val, y_val = methods.create_sequences(data_sc[num_train_samples:num_val_samples + num_train_samples], n_steps)
    
    
    # cross validation
    scores, avg_score = cross_validation(5, data = np.concatenate((x_train, x_val), axis = 0), labels = np.concatenate((y_train, y_val), axis = 0))
    
    print('Scores: ', scores)
    print('avg scores: ', avg_score)
    