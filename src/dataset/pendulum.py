"""
Created on Wed Jul 20 17:11:12 2022

@author: juanjo
"""

from numpy import sin, cos  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from collections import deque



class Pendulum(): 
    """ 
    
    Class for the double Pendulum system with the following conditions:
        
        1. The temporal step is always dt = 0.02
        2. The lengths, the masses and the gravity are always the same
        3. The only atribute is the lenght of the temporal vector
        
        """

    dt = 0.02 # time step
    
    def __init__(self, t_stop):
        self.t_stop = t_stop
        
    def derivs(self, state, t):
        """ defines the differential equations """
        
        G = 9.8  # acceleration due to gravity, in m/s^2
        L1 = 1.0  # length of pendulum 1 in m
        L2 = 1.0  # length of pendulum 2 in m
        L = L1 + L2  # maximal length of the combined pendulum
        M1 = 1.0  # mass of pendulum 1 in kg
        M2 = 1.0  # mass of pendulum 2 in kg
        
        dydx = np.zeros_like(state)
        dydx[0] = state[1] # Esta ecuación nos da theta1

        delta = state[2] - state[0]
        den1 = (M1+M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
        dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                    + M2 * G * sin(state[2]) * cos(delta)
                    + M2 * L2 * state[3] * state[3] * sin(delta)
                    - (M1+M2) * G * sin(state[0]))
                   / den1) # Esta ec. nos da vel1

        dydx[2] = state[3] # Esta ec. nos da theta2

        den2 = (L2/L1) * den1
        dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                    + (M1+M2) * G * sin(state[0]) * cos(delta)
                    - (M1+M2) * L1 * state[1] * state[1] * sin(delta)
                    - (M1+M2) * G * sin(state[2]))
                   / den2) # Esta ec nos da vel2

        return dydx
    
    def solver(self, state):
        """ Solves the differential equations """
        
        dt = 0.02
        t = np.arange(0, self.t_stop, dt)
        y = integrate.odeint(self.derivs, state, t)
        data = pd.DataFrame({'x1(Ang)': y[:, 0], 'x2(Vel1)': y[:, 1], 'x3(Ang2)': y[:, 2], 'x4(Vel2)': y[:, 3]})
            
        return data
        
    def ani(self, state):
        """ Generates the double pendulum animation """
            
        dt = 0.02
        L1 = 1.0
        L2 = 1.0
        
        data = self.solver(state)
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

                
        x1 = L1*np.sin(data.values[:, 0])
        y1 = -L1*np.cos(data.values[:, 0])

        x2 = L2*np.sin(data.values[:, 2]) + x1
        y2 = -L2*np.cos(data.values[:, 2]) + y1
            
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
            
        ani = animation.FuncAnimation(fig8, animate, len(y2), interval = dt*1000, blit = True)
        ani.save('pendulo.gif')
        
    def calculate_angular_velocities(self):
        """ Calculates the angular velocity """
        
        data = self.solver()
        return data[:, 1], data[:, 3]

    def calculate_kinetic_energy(self):
        """ Calculates the Kinetic Energy """
        G = 9.8  # acceleration due to gravity, in m/s^2
        l1 = 1.0  # length of pendulum 1 in m
        l2 = 1.0  # length of pendulum 2 in m
        l = l1 + l2  # maximal length of the combined pendulum
        m1 = 1.0  # mass of pendulum 1 in kg
        m2 = 1.0  # mass of pendulum 2 in kg
        ec = []
        data = self.solver()
        for dato in data.values():
            x1 = dato[0]
            v1 = dato[1]
            x2 = dato[2]
            v2 = dato[3]
            
            ec.append(0.5*m1*(v1*l1)**2 + 0.5*m2*((v1*l1)**2 + (v2*l2)**2 + 2.0*v1*v2*l1*l2*np.cos(x1-x2)))
        return ec
    
    def calculate_pot_energy(self):
        """ Calculates the potential energy """
        G = 9.8  # acceleration due to gravity, in m/s^2
        l1 = 1.0  # length of pendulum 1 in m
        l2 = 1.0  # length of pendulum 2 in m
        l =l1 + l2  # maximal length of the combined pendulum
        m1 = 1.0  # mass of pendulum 1 in kg
        m2 = 1.0  # mass of pendulum 2 in kg
        data = self.solver()
        v = []
        
        for dato in data.values():
            x1 = dato[0]
            v1 = dato[1]
            x2 = dato[2]
            v2 = dato[3]
            
            v.append(-m1*G*l1*np.cos(x1) - m2*G*(l1*np.cos(x1) + l2*np.cos(x2)))
        
        return v
    
    def calculate_total_energy(self):
        """ Calculates the total energy """
        v = self.CalculatePotEnergy()
        k = self.CalculateKEnergy()
        
        return [v[i] + k[i] for i in range(len(v))]
    
#    def T_max(self):
        """ Calculates the max period """
   
# Use case
if __name__ == "__main__":
    state = [0.02, 0.0, 0.0, 0.03]
    t_stop = 20
    p = Pendulum(state, t_stop)
    p.ani()