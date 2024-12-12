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
from collections import deque
import matplotlib.animation as animation

import utils.math_utils as math_utils
import dataset.pendulum as pendulum
import utils.methods as methods
from models.RNN import RNNmodel

n_steps = 30

# generate the data
state = [np.random.rand()*np.pi, np.random.rand()*np.pi, np.random.rand()*np.pi, np.random.rand()*np.pi]
t_stop = 20.0

p = pendulum.Pendulum(t_stop)
data = p.solver(state)
t = np.arange(0, t_stop, 0.02)

# split and scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_sc = scaler.fit_transform(data)

n = int(0.3*len(data))
data_pred = data_sc[:n]
data_test = data_sc[n:]

# load the RNN model
#RNNmodel = keras.models.load_model("RNNmodel.h5", custom_objects={'RNNmodel': RNNmodel})
RNNmodel = keras.models.load_model("RNNmodel.keras")

# predict the future and unscale the data
for i in range(len(data_test)):
    x_input = data_pred[i:i + n_steps]
    x_input = x_input.reshape((1, x_input.shape[0], x_input.shape[1]))
    predicted = RNNmodel.predict(x_input, verbose = 0)
    data_pred = np.concatenate((data_pred, predicted), axis = 0)
new_data = scaler.inverse_transform(data_pred)

"""
# plot
k = 0
plt.plot(t, new_data[:, k], label = "Prediction")
plt.plot(t, data.values[:, k], label = "Actual data")
plt.legend()
plt.title(r'$\theta$')
plt.show()
"""

# generate the animation
def ani(data, state, name):
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
     ani.save(name)
     
if __name__ == "__main__":
    ani(data.values, state, 'pendulo_true.gif')
    ani(new_data, state, 'pendulo_RNN.gif')