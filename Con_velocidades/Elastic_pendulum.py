#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 22:43:06 2022

@author: juanjo
"""

from Functions import separar_set, crear_ANN
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

""" Definimos la funcion para resolver las ecuaciones diferenciales """

g = 9.8  # acceleration due to gravity, in m/s^2
l1 = 1.0  # length of pendulum 1 in m
l2 = 1.0  # length of pendulum 2 in m
l = l1 + l2  # maximal length of the combined pendulum
m1 = 1.0  # mass of pendulum 1 in kg
m2 = 1.0  # mass of pendulum 2 in kg
k1 = 0.5
k2 = 0.6
t_stop = 15 # how many seconds to simulate
dt = 0.02 # El salto temporal
t = np.arange(0, t_stop, dt)

def Elastic_pendulum(s, t):
    dydx = np.zeros_like(s)
    
    dydx[0] = s[1] # theta1
    
    delta = s[0] - s[2]
    
    dydx[1] = (1.0/((m1+m2)*s[4]**2))*(-m2*s[5]*s[7]*np.sin(delta) 
    + m2*s[5]*s[6]*s[2]*np.cos(delta) - m2*s[7]*s[4]*s[1]*np.cos(-delta)
    - m2*s[4]*s[1]*s[6]*s[3]*np.sin(delta) + (m1+m2)*g*s[4]*np.sin(s[0]))
    # Vel1
    
    dydx[2] = s[3] # theta2
    
    dydx[3] = (1.0/(s[6]*s[6]))*(s[5]*s[7]*np.sin(delta) 
    - s[5]*s[6]*np.sin(delta) + s[5]*s[6]*s[2]*np.cos(delta)
    - s[6]*s[4]*s[1]*np.cos(delta) - s[7]*s[4]*s[1]*np.cos(-delta)
    + s[4]*s[6]*s[1]*s[3]*np.sin(delta) + g*s[6]*np.sin(s[2]))
    
    dydx[4] = s[5]
    
    dydx[5] = (1.0/(m1+m2))*(m1*s[4]*s[1]**2 + m2*s[4]*s[3]**2 +
        s[7]*s[1]*np.sin(-delta)*m2 - m1*g*np.cos(s[0])
        -m2*g*np.cos(s[2]) + k1*(s[4]-l1))
    
    dydx[6] = s[7]
    
    dydx[7] = s[6]*s[3]**2 + s[5]*s[2]*np.sin(delta) 
    + s[1]*s[6]*s[2]*np.cos(delta) - g*np.cos(s[2] + 
       (k2/m2)*(s[6] - l2))
    
    return dydx
    