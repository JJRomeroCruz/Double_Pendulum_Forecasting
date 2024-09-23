# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:21:36 2024

@author: Juanjo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import utils.math_utils as math_utils
import dataset.pendulum as pendulum
import utils.methods as methods
import models.RNN as RNN

# generate the dataset
state = [np.random.rand()*np.pi, np.random.rand()*np.pi, np.random.rand()*np.pi, np.random.rand()*np.pi]
t_stop = 500.0
dt = 0.02
t = np.arange(0, t_stop, dt)

p = pendulum.Pendulum(t_stop)
data = p.solver(state)

#p.ani(state)

print('Animacion hecha')

# split the dataframe in training, test and validation datasets
num_train_samples = int(0.5*len(data))
num_val_samples = int(0.25*len(data))
num_test_samples = len(data) - num_train_samples - num_val_samples

# scale the data
scaler = MinMaxScaler(feature_range = (0, 1))
data_sc = scaler.fit_transform(data)

# sequence the data
n_steps = 10

x_train, y_train = methods.create_sequences(data_sc[:num_train_samples], n_steps)
x_val, y_val = methods.create_sequences(data_sc[num_train_samples:num_val_samples + num_train_samples], n_steps)
x_test, y_test = methods.create_sequences(data_sc[num_val_samples + num_train_samples:], n_steps)

# define the model and train it
model = RNN.RNNmodel(input_shape = (x_train.shape[0], x_train.shape[1]), units = 200, output_size = 4)
model.compile(optimizer = 'adam', loss = 'mse')
model.summary()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 4))
y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))

x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 4))
y_val = y_val.reshape((y_val.shape[0], 1, y_val.shape[1]))

history = model.fit(x_train, y_train, shuffle = False, epochs = 20, batch_size = 32, validation_data = (x_val, y_val))

# plot the metrics
plt.plot(history.history['loss'], label = 'training loss')
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.legend()

# save the model
model.save("RNNmodel.h5")
