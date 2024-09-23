# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:38:41 2024

@author: Juanjo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataset.pendulum as pendulum
import utils.methods as methods
import utils.math_utils as math_utils

# Random initial conditions
#state = [np.random.rand()*np.pi, 0.0, np.random.rand()*np.pi, 0.0]
state = np.radians([-100.0, 4.0, 0.0, 90])
t_stop = 20.0

# Data
p = pendulum.Pendulum(t_stop)
data = p.solver(state)
t = np.linspace(0, t_stop, int(t_stop/0.02))

# Animation
p.ani(state)
#plt.plot(t, data.values[:, 0])

