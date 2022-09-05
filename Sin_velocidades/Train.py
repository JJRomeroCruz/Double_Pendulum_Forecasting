#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:01:51 2022

@author: juanjo
"""

from Functions import separar_set, crear_ANN
from Pendulum import derivs
from numpy import sin, cos  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from collections import deque
from statsmodels.tsa.arima.model import ARIMA 
from statsmodels.tsa.vector_ar.var_model import VAR

import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

m = 500
t_stop = 10 # how many seconds to simulate
dt = 0.02 # El salto temporal
t = np.arange(0, t_stop, dt)

""" Entrenamos la red neuronal """
taza_aprendizaje, taza_abandono = 0.00124108, 0.5
model_ANN = crear_ANN(taza_aprendizaje, taza_abandono)
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
    
history = model_ANN.fit(xTrain, yTrain, epochs=15, validation_data = (xTest, yTest))
model_ANN.evaluate(xTest, yTest)    

""" Sacamos la gráfica de la accuracy respecto de las épocas """
pd.DataFrame(history.history).plot(figsize = (8, 5))
plt.grid(True)
#plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

model_ANN.save("modelo_definitivo11.h5")