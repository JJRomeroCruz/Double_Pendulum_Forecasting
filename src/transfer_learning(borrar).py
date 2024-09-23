#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 18:21:24 2022

@author: juanjo
"""

""" Vamos a usar el transfer learning para el modelo con velocidades, va a consistir en 
coger el modelo de ANN para predecir la serie temporal con m = 1000, y vamos a añadirle una 
o dos capas más y entrenarlas con fine-tunning, a ver que tal """

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

""" Antes de nada, tenemos que generar el dataset, resolviendo m veces el sistema 
de ecuaciones diferenciales """
m = 1
t_stop = 50000 # how many seconds to simulate
dt = 0.02 # El salto temporal
t = np.arange(0, t_stop, dt)

""" Sacamos el dataset """
i = 0
xTrain, yTrain, xTest, yTest = np.empty((0, 4)), np.empty((0, 4)), np.empty((0, 4)), np.empty((0, 4))
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
    #datos = datos.drop(['x2(Vel1)', 'x4(Vel2)'], axis = 1)
    r = 0.8
    n = len(y)
    xTrain_ind, xTest_ind, yTrain_ind, yTest_ind = separar_set(datos.values[0:n-1], datos.values[1:n], n, r)
    
    xTrain = np.concatenate((xTrain, xTrain_ind), axis = 0)
    xTest = np.concatenate((xTest, xTest_ind), axis = 0)
    yTrain = np.concatenate((yTrain, yTrain_ind), axis = 0)
    yTest = np.concatenate((yTest, yTest_ind), axis = 0)
    i += 1

""" Ahora, procedemos a la transferencia del aprendizaje """

# Pasamos las capas del modelo 3 al 4 y añadimos al 4 las capas que queriamos
model3 = keras.models.load_model("modelo_definitivo5.h5")
model4 = keras.models.Sequential(model3.layers[:-1])
model4.add(keras.layers.Dense(500, activation = "selu", kernel_initializer = "lecun_normal")) # Añadimos una capa oculta más
#model4.add(keras.layers.BatchNormalization())
model4.add(Dense(500, activation = "selu", kernel_initializer = "lecun_normal"))
#model4.add(keras.layers.BatchNormalization()) # y le añadimos un batch normalization
model4.add(keras.layers.Dense(4)) # Capa de output

learning_rate = 0.001391702 # Es la tasa de aprendizaje más óptima que tenía el modelo 3
optimizer = keras.optimizers.SGD(learning_rate)

# Clonamos el modelo 3 para evitar que modifiquen dicho modelo
model3_clone = keras.models.clone_model(model3)
model3_clone.set_weights(model3.get_weights())

# Congelamos las capas heredadas del modelo 3
for layer in model4.layers[:-2]: # Le etoy diciendo que vaya desde la primera capa hasta dos capas antes del final
    layer.trainable = False
    
# Compilamos el modelo
model4.compile(loss = "mse", optimizer = optimizer, metrics = ["accuracy"])

# Entrenamos el modelo unas pocas epocas para que se acerquen un poco los pesos
history = model4.fit(xTrain, yTrain, epochs = 2, validation_data = (xTest, yTest))

# Descongelamos las capas para poder entrenarlas un poco ahora
for layer in model4.layers[:-2]:
    layer.trainable = True

# Volvemos a compilar el modelo ahora que las capas son entrenables, y los ajustamos esta vez de verdad
model4.compile(loss = "mse", optimizer = optimizer, metrics = ["accuracy"])
history = model4.fit(xTrain, yTrain, epochs = 5, validation_data = (xTest, yTest))
model4.evaluate(xTest, yTest)

""" Ahora, guardamos el modelo """
model4.save("modelo_definitivo6.h5")
