# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:31:15 2024

@author: Juanjo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import matplotlib.animation as animation

import dataset.pendulum as pendulum
import utils.methods as methods
import utils.math_utils as math_utils

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN, Dropout


# Animation with the original pendulum
dt = 0.02
state = np.radians([80.0, -20.0, -70.0, 100.0])
t_stop = 20.0
p = pendulum.Pendulum(t_stop)
data = p.solver(state)
t = np.arange(0, t_stop, dt)
#p.ani(state)

# Define the RNN model
y_train, y_test, t_train, t_test = methods.split_set(data, t, 0.3)

# Scale the data
scaler = MinMaxScaler(feature_range = (0, 1))
df_scaled = scaler.fit_transform(data)

# Creamos las secuencias para la red LSTM
def create_sequences(data, n_steps):
    x, y = [], []
    for i in range(len(data) - n_steps):
        x.append(data[i: i +n_steps])
        y.append(data[i + n_steps])
    return np.array(x), np.array(y)

n_steps = 50 # Numero de pasos de la secuencia
x, y = create_sequences(df_scaled, n_steps)

# Define the LSTM model
model = Sequential()
model.add(SimpleRNN(units = 40, activation = 'tanh', use_bias = True, input_shape=(x.shape[1], x.shape[2])))
#model.add(LSTM(units=100, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
#model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
#model.add(SimpleRNN(units = 100))
#model.add(Dense(50, activation = 'relu'))
"""
model.add(LSTM(units=100, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))  # Regularización para prevenir sobreajuste
model.add(LSTM(units=50))
model.add(Dense(10, activation='relu'))  # Capa intermedia
"""
model.add(Dense(4))

model.compile(optimizer = 'adam', loss = 'mse')

# Reshape de los datos para la LSTM [samples, time steps, features]
x = x.reshape((x.shape[0], x.shape[1], 4))
y = y.reshape((y.shape[0], 1, y.shape[1]))
# Entrenar el modelo
model.fit(x, y, epochs = 15, batch_size = 32, verbose = 0)

# prediction
df_scaled.shape
y_train_sc = scaler.fit_transform(y_train)
for i in range(len(y_test)):
    x_input = y_train_sc[i:i + n_steps].reshape((1, n_steps, 4))
    predicted = model.predict(x_input, verbose = 0)
    y_train_sc = np.concatenate((y_train_sc, predicted), axis = 0)
    
data_new = scaler.inverse_transform(y_train_sc)

# Animation
def ani(data, state):
     """ Generates the double pendulum animation """
         
     dt = 0.02
     L1 = 1.0
     L2 = 1.0
     
     #data = self.solver(state)
     L = L1+L2
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

             
     x1 = L1*np.sin(data[:, 0])
     y1 = -L1*np.cos(data[:, 0])

     x2 = L2*np.sin(data[:, 2]) + x1
     y2 = -L2*np.cos(data[:, 2]) + y1
         
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
         
     ani = animation.FuncAnimation(fig8, animate, len(data), interval = dt*1000, blit = True)
     ani.save('pendulo_RNN.gif')
     
ani(data_new, state)