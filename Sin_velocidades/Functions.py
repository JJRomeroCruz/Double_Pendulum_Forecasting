#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:17:38 2022

@author: juanjo
"""

""" This script contains the functions I use to forecast time series """
from numpy import sin, cos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import statsmodels.tsa.api as smt
from collections import deque
from statsmodels.tsa.arima.model import ARIMA 
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy import signal
from statsmodels.tsa.stattools import adfuller
import tensorflow.keras as keras


import seaborn as sns
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

def separar_set(x, y, n, r):
    
    """ Funcion para separar dos datasets: x, y; de dim n, en una 
    proporción r"""
    
    xTrain, xTest, yTrain, yTest = [], [], [], []
    for i in range(len(x)):
        if (i<=r*float(n)):
            xTrain.append(x[i])
            yTrain.append(y[i])
        else:
            xTest.append(x[i])
            yTest.append(y[i])
    return np.array(xTrain), np.array(xTest), np.array(yTrain), np.array(yTest)


def arima_univariant(datos, p, d, q):
    
    """ Funcion que nos define el modelo arima y nos predice una serie temporal en multipaso (todo de una vez)
    a partir de la mitad del dataframe datos, y con unos coeficientes p, d, q"""
    n = len(datos)
    # Separamos el dataset
    xTrain, yTrain, xTest, yTest = separar_set(datos, datos, n, 0.5)

    # Definimos los modelos ARIMA
    arima_model = ARIMA(xTrain, order = (p, d, q))
    model = arima_model.fit()

    y_pred = np.array(model.forecast(len(xTrain)))

    theta = np.concatenate((xTrain, y_pred), axis = None)
    
    return theta

def arma_univariant(datos):
    """ Funcion que nos define el modelo arma y nos predice una serie temporal en multipaso (todo de una vez)
    a partir de la mitad del dataframe datos, y con unos coeficientes p, d, q que se obtienen con el criterio AIC"""
    n = len(datos)
    best_aic = np.inf
    best_order = None
    best_mdl = None
    
    rng = range(5)
    
    for i in rng:
        for j in rng:
            try:
                tmp_mdl = smt.ARMA(datos, order = (i, j)).fit(method = 'mle', trend = 'nc')
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, j)
                    best_mdl = tmp_mdl
            except: continue
    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    # Separamos el dataset
    xTrain, yTrain, xTest, yTest = separar_set(datos, datos, n, 0.5)

    # Definimos los modelos ARIMA
    arma_model = smt.ARMA(xTrain, order = best_order)
    model = arma_model.fit()

    y_pred = np.array(model.forecast(len(xTrain)))

    theta = np.concatenate((xTrain, y_pred), axis = None)
    
    return theta

def test_estacionariedad(datos):
    """ Esta función nos dará una serie de parámetros para ver si una serie temporal
    es estacionaria o no, se va a usar el test de Dickey-Fuller, pero podríamos
    haber usado otro test como el KPSS ó el PP """
    
    print('Resultados del test de Dickey-Fuller: ')
    dftest = adfuller(datos, autolag = 't-stat')
    dfoutput = pd.Series(dftest[0:4], index = ['Test estadístico', 'p-valor', 'numero de lags usados', 'Numero de observaciones usadas'])
    for key, value in dftest[4].items():
        dfoutput['Valor critico(%s)'%key] = value
    dfoutput.to_csv('test_DF_' + str(datos[0]) + '.csv')
    print(dfoutput)

def var_multi(datos):
    
    """ Funcion que nos define el modelo VAR para series temporales, y nos 
    predice la serie temporal en multipaso (es decir, de una vez)"""
    
    y_verdad, yTrain, xTest, yTest = separar_set(datos, datos, len(datos), 0.5)

    model = VAR(endog = datos, exog = datos)
    model_fit = model.fit()
    
    y_pred = model_fit.forecast(y_verdad, exog_future = yTest, steps =  len(y_verdad[:len(yTest)]))

    y = np.concatenate((y_verdad, y_pred), axis = 0)
    return y        
    

def var_paso(datos):
    y_verdad, yTrain, xTest, yTest = separar_set(datos, datos, len(datos), 0.5)
    model = VAR(endog = datos, exog = datos)
    model_fit = model.fit()
    
    y_pred = np.empty((0, 4))
    while(len(y_verdad)<=len(datos)):
        n = len(y_verdad)
        y_pred = model_fit.forecast(y_verdad[n-1:n], exog_future = y_verdad[n-1:n], steps = 1)
        y_verdad = np.concatenate((y_verdad, y_pred), axis = 0)
    
    return y_verdad

    
     
    
""" Funcion que nos define el modelo VAR para series temporales, y nos 
    predice la serie temporal paso a paso (es decir, predice un punto, añade ese punto
    al dataset, y después predices el siguiente con el dataset resultante)""" 
    

def crear_ANN(taza_aprendizaje, taza_abandono):
    """ Funcion con la que definimos el modelo de red neuronal artificial """
    
    # Creamos el modelo
    model = Sequential()
    
    # Agregamos capas
    model.add(Dense(1000, input_dim = 2, activation = 'selu'))
    model.add(keras.layers.AlphaDropout(taza_abandono))
    
    model.add(Dense(350, activation = 'selu', kernel_initializer = "lecun_normal"))
    #model.add(keras.layers.AlphaDropout(taza_abandono))
    model.add(keras.layers.BatchNormalization())
    
    model.add(Dense(350, activation = 'selu', kernel_initializer = "lecun_normal"))
    #model.add(keras.layers.AlphaDropout(taza_abandono))
    model.add(keras.layers.BatchNormalization())
    
    model.add(Dense(350, activation = 'selu', kernel_initializer = "lecun_normal"))
    #model.add(keras.layers.AlphaDropout(taza_abandono))
    model.add(keras.layers.BatchNormalization())
    
    model.add(Dense(350, activation = 'selu', kernel_initializer = "lecun_normal"))
    #model.add(keras.layers.AlphaDropout(taza_abandono))
    model.add(keras.layers.BatchNormalization())
    
    #model.add(Dense(400, activation = 'selu', kernel_initializer = "lecun_normal"))
    #model.add(keras.layers.AlphaDropout(taza_abandono))
    #model.add(keras.layers.BatchNormalization())
    model.add(Dense(2))
    
    # Compilamos el modelo
    #adam = Adam(learning_rate = taza_aprendizaje)
    #model.compile(optimizer = adam, metrics = ['accuracy'], loss = 'mean_squared_error')
    optimizer = keras.optimizers.SGD(taza_aprendizaje)
    model.compile(loss = "mse", optimizer = optimizer, metrics = ['accuracy'])
    return model

def ANN_paso(model, datos):
    
    """ Funcion que nos permite predecir una serie temporal paso a paso
    con el modelo de red neuronal model, a partir de un dataframe datos"""
    y_verdad, yTrain, xTest, yTest = separar_set(datos, datos, len(datos), 0.5)

    y_pred = 0.0
    
    while(len(y_verdad)<len(datos)):
        i = len(y_verdad)
        y_pred = model(y_verdad[i-1:i])
        y_verdad = np.concatenate((y_verdad, y_pred), axis = 0)
    return y_verdad

def ANN_multipaso(model, datos):
    
    """ Funcion aue nos permite predecir la serie temporal con el modelo de red 
    neuronal model"""
    y_verdad, yTrain, xTest, yTest = separar_set(datos, datos, len(datos), 0.5)
    y_pred = model.predict(y_verdad)
    y_verdad = np.concatenate((y_verdad, y_pred), axis = 0)
    
    return y_verdad

def cuadratic_error(y_pred, y_real):
    error = 0.0
    for i in range(len(y_real)):
        error += (y_pred[i] - y_real[i])**2
    return np.sqrt(error)

def error(y_pred, y_real):
    error = []
    for i in range(len(y_real)):
        error.append((y_pred[i] - y_real[i])**2)
    return np.array(error)

def CalcularVelocidad(theta, t):
    """ Nos calcula la derivada discretizada de un vector theta
    respecto de t"""
    v = []
    for i in range(len(t)-1):
        v.append((theta[i+1]-theta[i])/(t[i+1]-t[i]))
    return np.array(v)

def CalcularECinetica(x1, x2, v1, v2, m1, m2, l1, l2, t):
    """ Funcion que nos calcula la energia cinetica del doble pendulo"""
    
    K = []
    for i in range(len(v1)):
        K.append(0.5*m1*(v1[i]*l1)**2 + 0.5*m2*((x1[i]*l1)**2 + (x2[i]*l2)**2 + 2.0*x1[i]*x2[i]*l1*l2*np.cos(x1[i]-x2[i])))
    return np.array(K)

def CalcularEPotencial(x1, x2, v1, v2, m1, m2, l1, l2, t):
    """ Funcion que nos calcula la energia potencial del doble pendulo"""
    Pot = []
    g = 9.807
    for i in range(len(v1)):
        Pot.append(-m1*g*l1*np.cos(x1[i]) - m2*g*(l1*np.cos(x1[i]) - l2*np.cos(x2[i])))
    return np.array(Pot)

def T_maximas(P, f):
    """ P es la potencia espectral y f la frecuencia """
    maximos = signal.argrelmax(P, axis = 0, order = 2)
    Pmaximos = []
    for i in maximos:
        Pmaximos.append(1.0/f[i])
    return np.array(Pmaximos)
            