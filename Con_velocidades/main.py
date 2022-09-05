#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:34:09 2022

@author: juanjo
"""

""" Esta es el main script del programa"""
from Functions import separar_set, arima_univariant, var_multi, crear_ANN, ANN_multipaso, cuadratic_error, error
from Functions import CalcularVelocidad, CalcularECinetica, CalcularEPotencial, T_maximas
from Functions import arma_univariant, test_estacionariedad
from Pendulum import derivs
from mpl_toolkits import mplot3d
from numpy import sin, cos  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from collections import deque
from statsmodels.tsa.arima.model import ARIMA 
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy import signal
import matplotlib.animation as animation
from collections import deque
import statsmodels.tsa.api as smt

import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

""" Definimos todas las constantes que vamos a necesitar y el tiempo"""

m = 1000
G = 9.8  # acceleration due to gravity, in m/s^2
L1 = 1.0  # length of pendulum 1 in m
L2 = 1.0  # length of pendulum 2 in m
L = L1 + L2  # maximal length of the combined pendulum
M1 = 1.0  # mass of pendulum 1 in kg
M2 = 1.0  # mass of pendulum 2 in kg
t_stop = 10 # how many seconds to simulate
dt = 0.02 # El salto temporal
t = np.arange(0, t_stop, dt)


""" Cargamos el modelo de red neuronal previamente entrenado"""
model_ANN = keras.models.load_model("modelo_definitivo3(Alt2).h5")


""" Resolvemos la ecuacion con las condiciones iniciales"""

th1 = -100.0
w1 = 30.0
th2 = -50.0
w2 = 0.0

# initial state
state = np.radians([th1, w1, th2, w2])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)
datos = pd.DataFrame({'x1(Ang1)': y[:, 0], 'x2(Vel1)': y[:, 1], 'x3(Ang2)': y[:, 2], 'x4(Vel2)': y[:, 3]})

# Volvemos a separar el dataset
r = 0.5
n = len(y)

#datos = datos_tiempo.drop(['x2(Vel1)', 'x4(Vel2)'], axis = 1)
#y_verdad, x_test, yTrain, yTest = separar_set(datos.values[0:n, :], datos.values[0:n, :], n, r)

# Separamos los datasets de cada particula por separado
xTrain1, yTrain1, xTest1, yTest1 = separar_set(datos.values[:, 0], datos.values[:, 0], n, 0.5)
xTrain2, yTrain2, xTest2, yTest2 = separar_set(datos.values[:, 1], datos.values[:, 1], n, 0.5)

""" Predecimos una mitad del dataset para cada método """

# Método ARIMA
p1, d1, q1 = 2, 2, 1
theta1_arima = arima_univariant(datos.values[:, 0], p1, d1, q1)
p2, d2, q2 = 2, 2, 1
omega1_arima = arima_univariant(datos.values[:, 1], p2, d2, q2)
p3, d3, q3 = 0, 2, 0
theta2_arima = arima_univariant(datos.values[:, 2], p3, d3, q3)
p4, d4, q4 = 1, 2, 2
omega2_arima = arima_univariant(datos.values[:, 3], p4, d4, q4)
datos_arima = pd.DataFrame({'x1(Ang1)': theta1_arima, 'x2(Vel1)': omega1_arima, 'x3(Ang2)': theta2_arima, 'x4(Vel2)': omega2_arima})


# Método VAR
y_var_multi = var_multi(datos.values)
datos_var = pd.DataFrame({'x1(Ang1)': y_var_multi[:, 0], 'x2(Vel1)': y_var_multi[:, 1], 'x3(Ang2)': y_var_multi[:, 2], 'x4(Vel2)': y_var_multi[:, 3]})

# Red neuronal
y_ANN_multipaso = ANN_multipaso(model_ANN, datos.values)
datos_ANN = pd.DataFrame({'x1(Ang1)': y_ANN_multipaso[:, 0], 'x2(Vel1)': y_ANN_multipaso[:, 1],'x3(Ang2)': y_ANN_multipaso[:, 2], 'x4(Vel2)': y_ANN_multipaso[:, 3]})


""" Sacamos el error cuadrático medio de las thetas para cada método con respecto al valor real"""

# Método ARIMA
error_ARIMA_theta1 = error(datos_arima.values[:, 0], datos.values[:, 0])
error_ARIMA_theta2 = error(datos_arima.values[:, 2], datos.values[:, 2])

# Método VAR 
error_var_theta1 = error(datos_var.values[:, 0], datos.values[:, 0])
error_var_theta2 = error(datos_var.values[:, 2], datos.values[:, 2])

error_ANN_theta1 = error(datos_ANN.values[:, 0], datos.values[:, 0])
error_ANN_theta2 = error(datos_ANN.values[:, 2], datos.values[:, 2])


""" Sacamos el error cuadrático medio de las omegas (velocidades) para cada método con respecto al valor real"""
 
# Método ARIMA
error_ARIMA_omega1 = error(datos_arima.values[:, 1], datos.values[:, 1])
error_ARIMA_omega2 = error(datos_arima.values[:, 3], datos.values[:, 3])

# Método VAR 
error_var_omega1 = error(datos_var.values[:, 1], datos.values[:, 1])
error_var_omega2 = error(datos_var.values[:, 3], datos.values[:, 3])

error_ANN_omega1 = error(datos_ANN.values[:, 1], datos.values[:, 1])
error_ANN_omega2 = error(datos_ANN.values[:, 3], datos.values[:, 3])

""" Sacamos la energia cinetica y potencial para cada dataset generado por cada metodo """
K_arima, V_arima = CalcularECinetica(datos_arima.values[:, 0], datos_arima.values[:, 2], datos_arima.values[:, 1], datos_arima.values[:, 3], M1, M2, L1, L2, t), CalcularEPotencial(datos_arima.values[:, 0], datos_arima.values[:, 2] , datos_arima.values[:, 1], datos_arima.values[:, 3], M1, M2, L1, L2, t)
K_var_multi, V_var_multi = CalcularECinetica(datos_var.values[:, 0], datos_var.values[:, 2], datos_var.values[:, 1], datos_var.values[:, 3], M1, M2, L1, L2, t), CalcularEPotencial(datos_var.values[:, 0], datos_var.values[:, 2], datos_var.values[:, 1], datos_var.values[:, 3], M1, M2, L1, L2, t)
K_ANN_multi, V_ANN_multi = CalcularECinetica(datos_ANN.values[:, 0], datos_ANN.values[:, 2], datos_ANN.values[:, 1], datos_ANN.values[:, 3], M1, M2, L1, L2, t), CalcularEPotencial(datos_ANN.values[:, 0], datos_ANN.values[:, 2], datos_ANN.values[:, 1], datos_ANN.values[:, 3], M1, M2, L1, L2, t)
K_true, V_true = CalcularECinetica(datos.values[:, 0], datos.values[:, 2], datos.values[:, 1], datos.values[:, 3], M1, M2, L1, L2, t), CalcularEPotencial(datos.values[:, 0], datos.values[:, 2], datos.values[:, 1], datos.values[:, 3], M1, M2, L1, L2, t)


""" Ploteamos las series temporales de las thetas (los angulos)"""
# Empezamos por la serie temporal de theta 1
fig1 = plt.figure(figsize=(15, 15))
for i in range(4):
    ax1 = fig1.add_subplot(2, 3, i+1)
    ax1.set_title("Grafico #%i"%int(i+1))
fig1.subplots_adjust(wspace = 0.2, hspace = 0.2)

# ARIMA univariante
fig1.axes[0].plot(t, datos_arima.values[:len(t), 0], label = "theta 1")
fig1.axes[0].plot(t, datos_arima.values[:len(t), 2], label = "theta 2")
fig1.axes[0].set_xlabel("Tiempo (s)")
fig1.axes[0].set_ylabel("Theta (rad)")
fig1.axes[0].set_title("ARIMA Uni")
fig1.axes[0].legend()

# VAR multistep
fig1.axes[1].plot(t, datos_var.values[:len(t), 0], label = "theta 1")
fig1.axes[1].plot(t, datos_var.values[:len(t), 2], label = "theta 2")
fig1.axes[1].set_title("VAR multistep")
fig1.axes[1].set_xlabel("Tiempo (s)")
fig1.axes[1].set_ylabel("Theta (rad)")
fig1.axes[1].legend()

# ANN multistep
fig1.axes[2].plot(t, datos_ANN.values[:len(t), 0], label = "theta 1")
fig1.axes[2].plot(t, datos_ANN.values[:len(t), 2], label = "theta 2")
fig1.axes[2].set_title("ANN multistep")
fig1.axes[2].set_xlabel("Tiempo (s)")
fig1.axes[2].set_ylabel("Theta (rad)")
fig1.axes[2].legend()

# Truth
fig1.axes[3].plot(t, datos.values[:len(t), 0], label = "theta 1")
fig1.axes[3].plot(t, datos.values[:len(t), 2], label = "theta 2")
fig1.axes[3].set_title("Truth")
fig1.axes[3].set_xlabel("Tiempo (s)")
fig1.axes[3].set_ylabel("Theta (rad)")
fig1.axes[3].legend()

fig1.savefig('thetas-t.pdf')

""" Sacamos las gráficas del error cuadratico medio de las thetas """
fig2 = plt.figure(figsize=(15, 15))
for i in range(3):
    ax2 = fig2.add_subplot(2, 3, i+1)
    ax2.set_title("Grafico #%i"%int(i+1))
fig2.subplots_adjust(wspace = 0.2, hspace = 0.2)

# ARIMA univariante
fig2.axes[0].plot(t, error_ARIMA_theta1, label = "theta 1")
fig2.axes[0].plot(t, error_ARIMA_theta2, label = "theta 2")
fig2.axes[0].set_title("ARIMA Uni")
fig2.axes[0].set_xlabel("Tiempo(s)")
fig2.axes[0].set_ylabel("Error cuadratico medio")
fig2.axes[0].legend()

# VAR multistep
fig2.axes[1].plot(t, error_var_theta1, label = "theta 1")
fig2.axes[1].plot(t, error_var_theta2, label = "theta 2")
fig2.axes[1].set_title("VAR multistep")
fig2.axes[1].set_xlabel("Tiempo(s)")
fig2.axes[1].set_ylabel("Error cuadratico medio")
fig2.axes[1].legend()

# ANN multistep
fig2.axes[2].plot(t, error_ANN_theta1, label = "theta 1")
fig2.axes[2].plot(t, error_ANN_theta2, label = "theta 2")
fig2.axes[2].set_title("ANN multistep")
fig2.axes[2].set_xlabel("Tiempo(s)")
fig2.axes[2].set_ylabel("Error cuadratico medio")
fig2.axes[2].legend()

fig2.savefig('errores_thetas.pdf')

""" Sacamos las graficas de las energias cinetica, potencial y total """
fig3 = plt.figure(figsize=(15, 15))
for i in range(4):
    ax3 = fig3.add_subplot(2, 3, i+1)
    ax3.set_title("Grafico #%i"%int(i+1))
fig3.subplots_adjust(wspace = 0.2, hspace = 0.2)

# ARIMA univariante
fig3.axes[0].plot(t, K_arima[:len(t)], label = "Cinetica")
fig3.axes[0].plot(t, V_arima[:len(t)], label = "Potencial")
fig3.axes[0].plot(t, K_arima[:len(t)]+V_arima[:len(t)], label = "Total")
fig3.axes[0].set_title("ARIMA Uni")
fig3.axes[0].set_xlabel("Tiempo (s)")
fig3.axes[0].set_ylabel("Energia (J)")
fig3.axes[0].legend()

# VAR multi
fig3.axes[1].plot(t, K_var_multi[:len(t)], label = "Cinetica")
fig3.axes[1].plot(t, V_var_multi[:len(t)], label = "Potencial")
fig3.axes[1].plot(t[:len(K_arima)], K_var_multi[:len(t)]+V_var_multi[:len(t)], label = "Total")
fig3.axes[1].set_title("Var multistep")
fig3.axes[1].set_xlabel("Tiempo (s)")
fig3.axes[1].set_ylabel("Energia (J)")
fig3.axes[1].legend()

# ANN multipaso
fig3.axes[2].plot(t, K_ANN_multi[:len(t)], label = "Cinetica")
fig3.axes[2].plot(t, V_ANN_multi[:len(t)], label = "Potencial")
fig3.axes[2].plot(t, V_ANN_multi[:len(t)] + K_ANN_multi[:len(t)], label = "Total")
fig3.axes[2].set_ylim(-20.0, 20.0)
fig3.axes[2].set_title("ANN multistep")
fig3.axes[2].set_xlabel("Tiempo (s)")
fig3.axes[2].set_ylabel("Energia (J)")
fig3.axes[2].legend()

# Truth
fig3.axes[3].plot(t, K_true[:len(t)], label = "Cinetica")
fig3.axes[3].plot(t, V_true[:len(t)], label = "Potencial")
fig3.axes[3].plot(t, K_true[:len(t)] + V_true[:len(t)], label = "Total")
#fig3.axes[5].set_ylim(-20.0, 20.0)
fig3.axes[3].set_title("True")
fig3.axes[3].set_xlabel("Tiempo (s)")
fig3.axes[3].set_ylabel("Energia (J)")
fig3.axes[3].legend()

fig3.savefig('energias.pdf')

""" Sacamos la densidad espectral mediante el periodograma de la serie temporal de los datos y los de la serie que 
se ha predecido con la red neuronal y lo representamos """

lent_mitad = int(0.5*len(t))
# Theta 1 vanilla
f_datos1, P_datos1 = signal.periodogram(datos.values[lent_mitad:, 0], fs=1.0)
# Theta 2 vanilla
f_datos2, P_datos2 = signal.periodogram(datos.values[lent_mitad:, 2], fs = 1.0)

# Theta 1 con la prediccion ANN
f_ANN1, P_ANN1 = signal.periodogram(y_ANN_multipaso[lent_mitad:, 0], fs = 4.0)
# Theta 2 con la prediccion ANN
f_ANN2, P_ANN2 = signal.periodogram(y_ANN_multipaso[lent_mitad:, 2], fs = 4.0)

# Theta 1 con la prediccion VAR
f_var1, P_var1 = signal.periodogram(y_var_multi[lent_mitad:, 0], fs = 1.0)
# Theta 2 con la prediccion VAR
f_var2, P_var2 = signal.periodogram(y_var_multi[lent_mitad:, 2], fs = 1.0)

# Theta 1 con la prediccion ARIMA
f_arima1, P_arima1 = signal.periodogram(theta1_arima, fs = 1.0)
# Theta 2 con la prediccion ARIMA
f_arima2, P_arima2 = signal.periodogram(theta2_arima, fs = 1.0)

# Ploteamos
fig4 = plt.figure(figsize=(15, 15))
for i in range(6):
    ax4 = fig4.add_subplot(2, 3, i+1)
    ax4.set_title("Grafico #%i"%int(i+1))
fig4.subplots_adjust(wspace = 0.2, hspace = 0.2)

# Vanilla vs ANN 1
fig4.axes[0].plot(f_datos1, P_datos1, label = 'Vanilla')
fig4.axes[0].plot(f_ANN1, P_ANN1, label = 'ANN')
fig4.axes[0].set_title("theta 1 (ANN)")
fig4.axes[0].legend()
fig4.axes[0].set_ylim(0, 2)
fig4.axes[0].set_xlim(0, 0.3)

# Vanilla vs ANN 2
fig4.axes[1].plot(f_datos2, P_datos2, label = 'Vanilla')
fig4.axes[1].plot(f_ANN2, P_ANN2, label = 'ANN')
fig4.axes[1].set_title("theta 2 (ANN)")
fig4.axes[1].legend()
fig4.axes[1].set_ylim(0, 2)
fig4.axes[1].set_xlim(0, 0.3)

# Vanilla vs arima 1
fig4.axes[2].plot(f_datos1, P_datos1, label = 'Vanilla')
fig4.axes[2].plot(f_arima1, P_arima1, label = 'ARIMA')
fig4.axes[2].set_title("theta 1 (ARIMA)")
fig4.axes[2].legend()
fig4.axes[2].set_ylim(0, 2)
fig4.axes[2].set_xlim(0, 0.3)

# Vanilla vs arima 2
fig4.axes[3].plot(f_datos1, P_datos1, label = 'Vanilla')
fig4.axes[3].plot(f_arima2, P_arima2, label = 'ARIMA')
fig4.axes[3].set_title("theta 2 (ARIMA)")
fig4.axes[3].legend()
fig4.axes[3].set_ylim(0, 2)
fig4.axes[3].set_xlim(0, 0.3)

# Vanilla vs VAR 1
fig4.axes[4].plot(f_datos1, P_datos1, label = 'Vanilla')
fig4.axes[4].plot(f_var1, P_var1, label = 'VAR')
fig4.axes[4].set_title("theta 1 (VAR)")
fig4.axes[4].legend()
fig4.axes[4].set_ylim(0, 2)
fig4.axes[4].set_xlim(0, 0.3)

# Vanilla vs VAR 2
fig4.axes[5].plot(f_datos1, P_datos1, label = 'Vanilla')
fig4.axes[5].plot(f_var2, P_var2, label = 'VAR')
fig4.axes[5].set_title("theta 2 (VAR)")
fig4.axes[5].legend()
fig4.axes[5].set_ylim(0, 2)
fig4.axes[5].set_xlim(0, 0.3)

fig4.savefig('periodogramas.pdf')

""" Ahora que hemos sacado los periodogramas, vamos a sacar los peaks, sus anchuras
 y los máximos para poder conocer los períodos de cada serie temporal """
 
# Vamos a empezar por los maximos para no fliparnos mucho
# ARIMA
Tmax_arima1 = T_maximas(P_arima1, f_arima1)
Tmax_arima2 = T_maximas(P_arima2, f_arima2)

# VAR
Tmax_var1 = T_maximas(P_var1, f_var1)
Tmax_var2 = T_maximas(P_var2, f_var2)

# True
Tmax_true1 = T_maximas(P_datos1, f_datos1)
Tmax_true2 = T_maximas(P_datos2, f_datos2)

# ANN
Tmax_ANN1 = T_maximas(P_ANN1, f_ANN1)
Tmax_ANN2 = T_maximas(P_ANN2, f_ANN2)


""" Vamos a sacar la densidad espectral por el metodo de welch en vez de por periodograma """

f = 1.0e-2
# Theta 1 vanilla
fw_datos1, Pw_datos1 = signal.welch(datos.values[lent_mitad:, 0], fs = 1.0)
# Theta 2 vanilla
fw_datos2, Pw_datos2 = signal.welch(datos.values[lent_mitad:, 2], fs = 1.0)

# Theta 1 ANN
fw_ANN1, Pw_ANN1 = signal.welch(y_ANN_multipaso[lent_mitad:, 0], fs = 1.0)
# Theta 2 ANN
fw_ANN2, Pw_ANN2 = signal.welch(y_ANN_multipaso[lent_mitad:, 2], fs = 1.0)

# Theta 1 VAR
fw_var1, Pw_var1 = signal.welch(y_var_multi[lent_mitad:, 0], fs = 1.0)
# Theta 2 VAR
fw_var2, Pw_var2 = signal.welch(y_var_multi[lent_mitad:, 2], fs = 1.0)

# Theta 1 ARIMA
fw_arima1, Pw_arima1 = signal.welch(theta1_arima[lent_mitad:], fs = 1.0)
# Theta 2 ARIMA
fw_arima2, Pw_arima2 = signal.welch(theta2_arima[lent_mitad:], fs = 1.0)

# Ploteamos
fig5 = plt.figure(figsize=(15, 15))
for i in range(6):
    ax5 = fig5.add_subplot(2, 3, i+1)
    ax5.set_title("Grafico #%i"%int(i+1))
# Vanilla vs ANN 1
fig5.axes[0].plot(fw_datos1, Pw_datos1, label = 'Vanilla')
fig5.axes[0].plot(fw_ANN1, Pw_ANN1, label = 'ANN')
fig5.axes[0].set_title("theta 1 (ANN)")
fig5.axes[0].legend()
fig5.axes[0].set_ylim(0, 2)
fig5.axes[0].set_xlim(0, 0.3)

# Vanilla vs ANN 2
fig5.axes[1].plot(fw_datos2, Pw_datos2, label = 'Vanilla')
fig5.axes[1].plot(fw_ANN2, Pw_ANN2, label = 'ANN')
fig5.axes[1].set_title("theta 2 (ANN)")
fig5.axes[1].legend()
fig5.axes[1].set_ylim(0, 2)
fig5.axes[1].set_xlim(0, 0.3)

# Vanilla vs arima 1
fig5.axes[2].plot(fw_datos1, Pw_datos1, label = 'Vanilla')
fig5.axes[2].plot(fw_arima1, Pw_arima1, label = 'ARIMA')
fig5.axes[2].set_title("theta 1 (ARIMA)")
fig5.axes[2].legend()
fig5.axes[2].set_ylim(0, 2)
fig5.axes[2].set_xlim(0, 0.3)

# Vanilla vs arima 2
fig5.axes[3].plot(fw_datos1, Pw_datos1, label = 'Vanilla')
fig5.axes[3].plot(fw_arima2, Pw_arima2, label = 'ARIMA')
fig5.axes[3].set_title("theta 2 (ARIMA)")
fig5.axes[3].legend()
fig5.axes[3].set_ylim(0, 2)
fig5.axes[3].set_xlim(0, 0.3)

# Vanilla vs VAR 1
fig5.axes[4].plot(fw_datos1, Pw_datos1, label = 'Vanilla')
fig5.axes[4].plot(fw_var1, Pw_var1, label = 'VAR')
fig5.axes[4].set_title("theta 1 (VAR)")
fig5.axes[4].legend()
fig5.axes[4].set_ylim(0, 2)
fig5.axes[4].set_xlim(0, 0.3)

# Vanilla vs VAR 2
fig5.axes[5].plot(fw_datos1, Pw_datos1, label = 'Vanilla')
fig5.axes[5].plot(fw_var2, Pw_var2, label = 'VAR')
fig5.axes[5].set_title("theta 2 (VAR)")
fig5.axes[5].legend()
fig5.axes[5].set_ylim(0, 2)
fig5.axes[5].set_xlim(0, 0.3)

fig5.savefig('welch.pdf')

""" Sacamos el espectrograma """

# theta 1 vanilla
fs_datos1, tiempo, Ss_datos1 = signal.spectrogram(datos.values[lent_mitad:, 0], fs = 1.0)
# theta 2 vanilla
fs_datos2, tiempo, Ss_datos2 = signal.spectrogram(datos.values[lent_mitad:, 2], fs = 1.0)

# theta 1 ANN
fs_ANN1, tiempo, Ss_ANN1 = signal.spectrogram(y_ANN_multipaso[lent_mitad:, 0], fs = 1.0)
# theta 2 ANN
fs_ANN2, tiempo, Ss_ANN2 = signal.spectrogram(y_ANN_multipaso[lent_mitad:, 2], fs = 1.0)

# Ploteamos en 3d
fig6 = plt.figure()
ax6 = plt.axes(projection = '3d')

#ax6.plot3D(fs_datos1, Ss_datos1[:, 0], Ss_datos1[:, 1], 'gray')
fig6.savefig('spectrogram.pdf')

""" Vamos a sacar la gráfica theta1-theta2 para cada método de prediccion """
fig7 = plt.figure(figsize=(15, 15))
for i in range(4):
    ax7 = fig7.add_subplot(2, 3, i+1)
    ax7.set_title("Grafico #%i"%int(i+1))
fig7.subplots_adjust(wspace = 0.2, hspace = 0.2)

# ARIMA
fig7.axes[0].plot(theta1_arima, theta2_arima)
fig7.axes[0].set_title("ARIMA Univariante")
fig7.axes[0].set_ylabel("$\theta 2$")
fig7.axes[0].set_xlabel("$\theta 1$")
fig7.axes[0].legend()

# VAR (siempre multistep)
fig7.axes[1].plot(y_var_multi[:, 0], y_var_multi[:, 2])
fig7.axes[1].set_title("VAR multistep")
fig7.axes[1].set_xlabel("$\theta 1$")
fig7.axes[1].set_ylabel("$\theta 2$")

# ANN (multistep)
fig7.axes[2].plot(y_ANN_multipaso[:, 0], y_ANN_multipaso[:, 2])
fig7.axes[2].set_xlabel("$\theta 1$")
fig7.axes[2].set_ylabel("$\theta 2$")
fig7.axes[2].set_title("ANN multistep")
fig7.axes[2].legend()

# Truth
fig7.axes[3].plot(datos.values[:, 0], datos.values[:, 2])
fig7.axes[3].set_xlabel("$\theta 1$")
fig7.axes[3].set_ylabel("$\theta 2$")
fig7.axes[3].set_title("Truth")
fig7.axes[3].legend()

fig7.savefig('theta1-theta2.pdf')

""" Comemzamos con las animaciones """
history_len = 500
# El true
fig8 = plt.figure(figsize = (5, 4))
ax8 = fig8.add_subplot(xlim = (-L, L), ylim = (-L, 1.))
ax8.set_aspect('equal')
ax8.grid()

line, = ax8.plot([], [], 'o-', lw = 2)
trace, = ax8.plot([], [], '.-', lw = 1, ms = 2)
time_template = 'time = %.1fs'
time_text = ax8.text(0.05, 0.9, '', transform = ax8.transAxes)
history_x, history_y = deque(maxlen = history_len), deque(maxlen = history_len)

x1 = L1*np.sin(datos.values[:, 0])
y1 = -L1*np.cos(datos.values[:, 0])

x2 = L2*np.sin(datos.values[:, 2]) + x1
y2 = -L2*np.cos(datos.values[:, 2]) + y1

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    if i==0:
        history_x.clear()
        history_y.clear()
    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])
    
    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template %(i*dt))
    return line, trace, time_text
ani = animation.FuncAnimation(fig8, animate, len(y), interval = dt*1000, blit = True)
ani.save('pendulo_True.gif')

# ANN (siempre multipaso)
fig9 = plt.figure(figsize = (5, 4))
ax9 = fig9.add_subplot(xlim = (-L, L), ylim = (-L, 1.))
ax9.set_aspect('equal')
ax9.grid()

line, = ax9.plot([], [], 'o-', lw = 2)
trace, = ax9.plot([], [], '.-', lw = 1, ms = 2)
time_template = 'time = %.1fs'
time_text = ax9.text(0.05, 0.9, '', transform = ax9.transAxes)
history_x, history_y = deque(maxlen = history_len), deque(maxlen = history_len)

x1 = L1*np.sin(y_ANN_multipaso[:, 0])
y1 = -L1*np.cos(y_ANN_multipaso[:, 0])

x2 = L2*np.sin(y_ANN_multipaso[:, 2]) + x1
y2 = -L2*np.cos(y_ANN_multipaso[:, 2]) + y1

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    if i==0:
        history_x.clear()
        history_y.clear()
    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])
    
    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template %(i*dt))
    return line, trace, time_text
ani = animation.FuncAnimation(fig9, animate, len(y), interval = dt*1000, blit = True)
ani.save('pendulo_ANN.gif')

# VAR multipaso
fig9 = plt.figure(figsize = (5, 4))
ax9 = fig9.add_subplot(xlim = (-L, L), ylim = (-L, 1.))
ax9.set_aspect('equal')
ax9.grid()

line, = ax9.plot([], [], 'o-', lw = 2)
trace, = ax9.plot([], [], '.-', lw = 1, ms = 2)
time_template = 'time = %.1fs'
time_text = ax9.text(0.05, 0.9, '', transform = ax9.transAxes)
history_x, history_y = deque(maxlen = history_len), deque(maxlen = history_len)

x1 = L1*np.sin(y_var_multi[:, 0])
y1 = -L1*np.cos(y_var_multi[:, 0])

x2 = L2*np.sin(y_var_multi[:, 2]) + x1
y2 = -L2*np.cos(y_var_multi[:, 2]) + y1

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    if i==0:
        history_x.clear()
        history_y.clear()
    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])
    
    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template %(i*dt))
    return line, trace, time_text
ani = animation.FuncAnimation(fig9, animate, len(y), interval = dt*1000, blit = True)
ani.save('pendulo_VAR.gif')

# ARIMA univariante
fig10 = plt.figure(figsize = (5, 4))
ax10 = fig10.add_subplot(xlim = (-L, L), ylim = (-L, 1.))
ax10.set_aspect('equal')
ax10.grid()

line, = ax10.plot([], [], 'o-', lw = 2)
trace, = ax10.plot([], [], '.-', lw = 1, ms = 2)
time_template = 'time = %.1fs'
time_text = ax10.text(0.05, 0.9, '', transform = ax10.transAxes)
history_x, history_y = deque(maxlen = history_len), deque(maxlen = history_len)

x1 = L1*np.sin(theta1_arima)
y1 = -L1*np.cos(theta1_arima)

x2 = L2*np.sin(theta2_arima) + x1
y2 = -L2*np.cos(theta2_arima) + y1

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    if i==0:
        history_x.clear()
        history_y.clear()
    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])
    
    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template %(i*dt))
    return line, trace, time_text
ani = animation.FuncAnimation(fig10, animate, len(y), interval = dt*1000, blit = True)
ani.save('pendulo_ARIMA.gif')

""" Vamos a sacar las gráficas de las medias móviles de cada solución """
fig11 = plt.figure(figsize=(15, 15))
for i in range(4):
    ax11 = fig11.add_subplot(2, 3, i+1)
    ax11.set_title("Grafico #%i"%int(i+1))
fig11.subplots_adjust(wspace = 0.2, hspace = 0.2)

# ARIMA
fig11.axes[0].plot(t, datos_arima['x1(Ang1)'].rolling(window = 10).mean()[:len(t)], label = "theta1")
fig11.axes[0].plot(t, datos_arima['x3(Ang2)'].rolling(window = 10).mean()[:len(t)], label = "theta2")
fig11.axes[0].set_title("ARIMA Univariante")
fig11.axes[0].set_ylabel("Media movil")
fig11.axes[0].set_xlabel("Tiempo(s)")
fig11.axes[0].legend()

# VAR (siempre multistep)
fig11.axes[1].plot(t, datos_var['x1(Ang1)'].rolling(window = 10).mean()[:len(t)], label = "theta1")
fig11.axes[1].plot(t, datos_var['x3(Ang2)'].rolling(window = 10).mean()[:len(t)], label = "theta2")
fig11.axes[1].set_title("VAR multistep")
fig11.axes[1].set_xlabel("Tiempo(s)")
fig11.axes[1].set_ylabel("Media movil")

# ANN (multistep)
fig11.axes[2].plot(t, datos_ANN['x1(Ang1)'].rolling(window = 10).mean()[:len(t)], label = "theta1")
fig11.axes[2].plot(t, datos_ANN['x3(Ang2)'].rolling(window = 10).mean()[:len(t)], label = "theta2")
fig11.axes[2].set_xlabel("Tiempo(s)")
fig11.axes[2].set_ylabel("Media movil")
fig11.axes[2].set_title("ANN multistep")
fig11.axes[2].legend()

# Truth
fig11.axes[3].plot(t, datos['x1(Ang1)'].rolling(window = 10).mean()[:len(t)], label = "theta1")
fig11.axes[3].plot(t, datos['x3(Ang2)'].rolling(window = 10).mean()[:len(t)], label = "theta2")
fig11.axes[3].set_xlabel("Tiempo(s)")
fig11.axes[3].set_ylabel("Media movil")
fig11.axes[3].set_title("Truth")
fig11.axes[3].legend()

fig11.savefig('Media_movil.pdf')

""" Vamos a sacar la autocovarianza de cada dataset """
fig12 = plt.figure(figsize=(15, 15))
for i in range(4):
    ax12 = fig12.add_subplot(2, 3, i+1)
    ax12.set_title("Grafico #%i"%int(i+1))
fig12.subplots_adjust(wspace = 0.2, hspace = 0.2)

# ARIMA
fig12.axes[0].plot(t, smt.stattools.acovf(datos_arima.values[:, 0])[:len(t)], label = "theta1")
fig12.axes[0].plot(t, smt.stattools.acovf(datos_arima.values[:, 2])[:len(t)], label = "theta2")
fig12.axes[0].set_title("ARIMA Univariante")
fig12.axes[0].set_ylabel("Autocovarianza")
fig12.axes[0].set_xlabel("Tiempo(s)")
fig12.axes[0].legend()

# VAR (siempre multistep)
fig12.axes[1].plot(t, smt.stattools.acovf(datos_var.values[:, 0])[:len(t)], label = "theta1")
fig12.axes[1].plot(t, smt.stattools.acovf(datos_var.values[:, 2])[:len(t)], label = "theta2")
fig12.axes[1].set_title("VAR multistep")
fig12.axes[1].set_xlabel("Tiempo(s)")
fig12.axes[1].set_ylabel("Autocovarianza")

# ANN (multistep)
fig12.axes[2].plot(t, smt.stattools.acovf(datos_ANN.values[:, 0])[:len(t)], label = "theta1")
fig12.axes[2].plot(t, smt.stattools.acovf(datos_ANN.values[:, 2])[:len(t)], label = "theta2")
fig12.axes[2].set_xlabel("Tiempo(s)")
fig12.axes[2].set_ylabel("Autocovarianza")
fig12.axes[2].set_title("ANN multistep")
fig12.axes[2].legend()

# Truth
fig12.axes[3].plot(t, smt.stattools.acovf(datos.values[:, 0])[:len(t)], label = "theta1")
fig12.axes[3].plot(t, smt.stattools.acovf(datos.values[:, 2])[:len(t)], label = "theta2")
fig12.axes[3].set_xlabel("Tiempo(s)")
fig12.axes[3].set_ylabel("Autocovarianza")
fig12.axes[3].set_title("Truth")
fig12.axes[3].legend()

fig12.savefig('Autocovarianza.pdf')

""" Sacamos las gráficas del error cuadratico medio de las omegas """
fig13 = plt.figure(figsize=(15, 15))
for i in range(3):
    ax13 = fig13.add_subplot(2, 3, i+1)
    ax13.set_title("Grafico #%i"%int(i+1))
fig13.subplots_adjust(wspace = 0.2, hspace = 0.2)

# ARIMA univariante
fig13.axes[0].plot(t, error_ARIMA_omega1[:len(t)], label = "omega 1")
fig13.axes[0].plot(t, error_ARIMA_omega2[:len(t)], label = "omega 2")
fig13.axes[0].set_title("ARIMA Uni")
fig13.axes[0].set_xlabel("Tiempo(s)")
fig13.axes[0].set_ylabel("Error cuadratico medio")
fig13.axes[0].legend()

# VAR multistep
fig13.axes[1].plot(t, error_var_omega1[:len(t)], label = "omega 1")
fig13.axes[1].plot(t, error_var_omega2[:len(t)], label = "omega 2")
fig13.axes[1].set_title("VAR multistep")
fig13.axes[1].set_xlabel("Tiempo(s)")
fig13.axes[1].set_ylabel("Error cuadratico medio")
fig13.axes[1].legend()

# ANN multistep
fig13.axes[2].plot(t, error_ANN_omega1[:len(t)], label = "omega 1")
fig13.axes[2].plot(t, error_ANN_omega2[:len(t)], label = "omega 2")
fig13.axes[2].set_title("ANN multistep")
fig13.axes[2].set_xlabel("Tiempo(s)")
fig13.axes[2].set_ylabel("Error cuadratico medio")
fig13.axes[2].legend()

fig13.savefig('errores_omegas.pdf')

""" Ploteamos las series temporales de las omegas (las velocidades angulares)"""
# Empezamos por la serie temporal de theta 1
fig14 = plt.figure(figsize=(15, 15))
for i in range(4):
    ax14 = fig1.add_subplot(2, 3, i+1)
    ax14.set_title("Grafico #%i"%int(i+1))
fig14.subplots_adjust(wspace = 0.2, hspace = 0.2)

n = len(datos_arima.values[:, 1])
# ARIMA univariante
fig14.axes[0].plot(t[:n], datos_arima.values[:, 1], label = "omega 1")
fig14.axes[0].plot(t[:n], datos_arima.values[:, 3], label = "omega 2")
fig14.axes[0].set_xlabel("Tiempo (s)")
fig14.axes[0].set_ylabel("Omega (rad/s)")
fig14.axes[0].set_title("ARIMA Uni")
fig14.axes[0].legend()

# VAR multistep
fig14.axes[1].plot(t[:n], datos_var.values[:, 1], label = "omega 1")
fig14.axes[1].plot(t[:n], datos_var.values[:, 3], label = "omega 2")
fig14.axes[1].set_title("VAR multistep")
fig14.axes[1].set_xlabel("Tiempo (s)")
fig14.axes[1].set_ylabel("Omega (rad/s)")
fig14.axes[1].legend()

# ANN multistep
fig14.axes[2].plot(t[:n], datos_ANN.values[:, 1], label = "omega 1")
fig14.axes[2].plot(t[:n], datos_ANN.values[:, 3], label = "omega 2")
fig14.axes[2].set_title("ANN multistep")
fig14.axes[2].set_xlabel("Tiempo (s)")
fig14.axes[2].set_ylabel("Omega (rad/s)")
fig14.axes[2].legend()

# Truth
fig14.axes[3].plot(t[:n], datos.values[:, 1], label = "omega 1")
fig14.axes[3].plot(t[:n], datos.values[:, 3], label = "omega 2")
fig14.axes[3].set_title("Truth")
fig14.axes[3].set_xlabel("Tiempo (s)")
fig14.axes[3].set_ylabel("Omega (rad/s)")
fig14.axes[3].legend()

fig14.savefig('omegas-t.pdf')


""" Vamos a hacer el test de estacionariedad de los datos """
test_estacionariedad(datos.values[:, 0])
test_estacionariedad(datos.values[:, 1])
test_estacionariedad(datos.values[:, 2])
test_estacionariedad(datos.values[:, 3])

plt.show()
