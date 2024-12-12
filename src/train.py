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
from cross_validation import cross_validation

# generate the dataset
state = [np.random.rand()*np.pi, np.random.rand()*np.pi, np.random.rand()*np.pi, np.random.rand()*np.pi]
t_stop = 2000.0
dt = 0.02
t = np.arange(0, t_stop, dt)

p = pendulum.Pendulum(t_stop)
data = p.solver(state)

#p.ani(state)

print('Animation done')

# split the dataframe in training, test and validation datasets
num_train_samples = int(0.5*len(data))
num_val_samples = int(0.25*len(data))
num_test_samples = len(data) - num_train_samples - num_val_samples

# scale the data
scaler = MinMaxScaler(feature_range = (0, 1))
data_sc = scaler.fit_transform(data)

# sequence the data
n_steps = 60

x_train, y_train = methods.create_sequences(data_sc[:num_train_samples], n_steps)
x_val, y_val = methods.create_sequences(data_sc[num_train_samples:num_val_samples + num_train_samples], n_steps)
x_test, y_test = methods.create_sequences(data_sc[num_val_samples + num_train_samples:], n_steps)

# Evaluate the model with cross validation
#score, avg_score = cross_validation(5, data = np.concatenate((x_train, x_val), axis = 0), labels = np.concatenate((y_train, y_val), axis = 0))


# define the model and train it
model = RNN.RNNmodel(input_shape = (x_train.shape[0], x_train.shape[1]), units = 40, output_size = 4)
model.compile(optimizer = 'adam', loss = 'mse')
model.summary()

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 4))
y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))

x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 4))
y_val = y_val.reshape((y_val.shape[0], 1, y_val.shape[1]))

history = model.fit(x_train, y_train, shuffle = False, epochs = 5, batch_size = 64, validation_data = (x_val, y_val))

# plot the metrics

plt.plot(history.history['loss'], label = 'training loss')
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.legend()

# Evaluate the final model 
#test_loss = model.evaluate(x_val, y_val, verbose = 0)
#print('test loss: ', test_loss)

# save the model
model.save("RNNmodel.keras")

