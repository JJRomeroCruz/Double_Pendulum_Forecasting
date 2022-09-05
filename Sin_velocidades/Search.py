#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:32:49 2022

@author: juanjo
"""

""" En este script vamos a buscar el modelo que mejor nos de una predicción de los datos
explorando el espacion de hiperparámetros en base a unas métricas mediante validación 
cruzada, Cristo en ti confío porque no sea mu largo esto """

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
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV


m = 1000
t_stop = 10 # how many seconds to simulate
dt = 0.02 # El salto temporal
t = np.arange(0, t_stop, dt)

""" Sacamos el dataset """
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

""" Ahora, procedemos a la búsqueda del mejor modelo """

# Creamos la funcion que construye y compila el modelo, dado un conjunto de hiperparámetros
def build_model(n_hidden, n_neurons, learning_rate, input_shape = [2]):
    model = Sequential()
    options = {"input_shape": input_shape}
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation = "selu", kernel_initializer = "lecun_normal", **options))
        model.add(keras.layers.BatchNormalization())
        options = {}
    # Metemos una capa dropout despues de la ultima capa oculta
    #model.add(keras.layers.Dropout(rate = 0.5))
    model.add(Dense(2, **options))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss = "mse", optimizer = optimizer)
    return model

# Creamos un KerasRegressor basado en la función que construye el modelo
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

# Ahora, vamos a ver qué combinación de hiperparámetros es lo más óptimo
params_distribs = {
        "n_hidden": [5, 6],
        "n_neurons": np.arange(380, 400),
        "learning_rate": reciprocal(9e-4, 7e-3),
        }
rnd_search_cv = RandomizedSearchCV(keras_reg, params_distribs, n_iter = 10, cv = 3)
rnd_search_cv.fit(xTrain, yTrain, epochs = 10, validation_data = (xTest, yTest), callbacks = [keras.callbacks.EarlyStopping(patience = 10)])

# Enseñamos los hiperparametros más óptimos, los pasamos a un fichero, y guardamos el modelo
cadena = "Best params: " + str(rnd_search_cv.best_params_) + "\n" + "Best score: " + str(rnd_search_cv.best_score_)
print(cadena)

model = rnd_search_cv.best_estimator_.model
model.save("modelo_definitivo5.h5")

file = open("params.txt", 'a', encoding = 'utf8')
file.write(cadena)
file.close()

