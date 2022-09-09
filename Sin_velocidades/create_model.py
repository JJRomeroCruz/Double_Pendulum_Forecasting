#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:49:57 2022

@author: juanjo
"""
import pandas as pd
import numpy as np 
import seaborn as sns
import tensorflow.keras as keras
import scipy.integrate as integrate
import tensorflow.keras.optimizers

from Functions import separar_set
from Pendulum import derivs
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

""" Definimos los parámetros """
m = 1000
t_stop = 10 # how many seconds to simulate
dt = 0.02 # El salto temporal
t = np.arange(0, t_stop, dt)


""" Creamos el modelo """
model = Sequential()
n_hidden = 4 # Iremos eligiendo el numero de capas ocultas
n_neurons = 380 # Lo mismo, vamos eligiendo
learning_rate = 0.00124108 # tasa de aprendizaje
model.add(keras.layers.Dense(n_neurons, input_shape = [2])) # Capa de input
for i in range(n_hidden):
    model.add(Dense(n_neurons, activation = "selu", kernel_initializer = "lecun_normal")) # Capas ocultas
    model.add(keras.layers.BatchNormalization())
model.add(Dense(2)) # Capa de output
optimizer = keras.optimizers.SGD(learning_rate)
model.compile(loss = "mse", optimizer = optimizer)


""" Esta forma de entrenar es la tercera, que se ha visto que no es muy buena """
"""
xTrain, yTrain, xTest, yTest = np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))
while(i<=m):
    
    # Generamos las condiciones inciales
    th1_train = 180.0*np.random.rand()
    w1_train = 180.0*np.random.rand()
    th2_train = 180.0*np.random.rand()
    w2_train = 180.0*np.random.rand()

    # initial state
    state = np.radians([th1_train, w1_train, th2_train, w2_train])

    # integrate your ODE using scipy.integrate.
    y = integrate.odeint(derivs, state, t)
    
    datos = pd.DataFrame({'x1(Ang1)': y[:, 0], 'x2(Ang2)': y[:, 2]})
    r = 0.8
    n = len(y)
    xTrain, xTest, yTrain, yTest = separar_set(datos.values[0:n-1], datos.values[1:n], n, r)
    model.fit(xTrain, yTrain, epochs = 6, validation_data = (xTest, yTest), callbacks = [keras.callbacks.EarlyStopping(patience = 10)])

    i += 1
"""

""" Esta es la primera forma de entrenar el modelo y es el que mejor ha funcionado """
i = 0
xTrain, yTrain, xTest, yTest = np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))
while(i<=m):
    
    # Generamos las condiciones inciales
    th1_train = 180.0*np.random.rand()
    w1_train = 180.0*np.random.rand()
    th2_train = 180.0*np.random.rand()
    w2_train = 180.0*np.random.rand()

    # initial state
    state = np.radians([th1_train, w1_train, th2_train, w2_train])

    # integrate your ODE using scipy.integrate.
    y = integrate.odeint(derivs, state, t)
    
    datos = pd.DataFrame({'x1(Ang1)': y[:, 0], 'x2(Vel1)': y[:, 1], 'x3(Ang2)': y[:, 2], 'x4(Vel2)': y[:, 3]})
    datos = datos.drop(['x2(Vel1)', 'x4(Vel2)'], axis = 1)
    r = 0.9
    n = len(y)
    xTrain_ind, xTest_ind, yTrain_ind, yTest_ind = separar_set(datos.values[0:n-1], datos.values[1:n], n, r)
    
    xTrain = np.concatenate((xTrain, xTrain_ind), axis = 0)
    xTest = np.concatenate((xTest, xTest_ind), axis = 0)
    yTrain = np.concatenate((yTrain, yTrain_ind), axis = 0)
    yTest = np.concatenate((yTest, yTest_ind), axis = 0)
    i += 1

""" Entrenamos el modelo con el dataset resultante """
model.fit(xTrain, yTrain, epochs = 10, validation_data = (xTest, yTest), callbacks = [keras.callbacks.EarlyStopping(patience = 10)])


model.save("modelo1.h5")