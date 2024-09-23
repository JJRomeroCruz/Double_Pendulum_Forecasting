# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 23:01:24 2024

@author: Juanjo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras

import utils.math_utils as math_utils
import dataset.pendulum as pendulum
import utils.methods as methods

# generate the data
state = [np.random.rand()*np.pi, np.random.rand()*np.pi, np.random.rand()*np.pi, np.random.rand()*np.pi]
t_stop = 40.0

p = pendulum.Pendulum(t_stop)
data = p.solver(state)
t = np.arange(0, t_stop, 0.02)

# split and scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_sc = scaler.fit_transform(data)

n = int(0.3*len(data))
data_pred = data_sc.values[:n]
data_test = data_sc.values[n:]

# load the RNN model
RNNmodel = keras.models.load_model("RNNmodel.h5")

# predict the future and unscale the data

# generate the animations