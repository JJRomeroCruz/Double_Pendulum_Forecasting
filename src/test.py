# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 10:47:42 2024

@author: Juanjo
"""

import pytest
import numpy as np
import pandas as pd

import prediction
import utils.methods as methods
import models.RNN as RNN
import train as train

# load the simulated data
@pytest.fixture
def data():
    return prediction.data
@pytest.fixture
def new_data():
    return prediction.new_data
@pytest.fixture
def history():
    return train.history

# Test Class
class Test_Prediction():
    def test_len_df(self, data):
        """ the lenght in the two both pendulum must be 1 """
        mod1 = [np.cos(x[0])**2 + np.sin(x[0])**2 for x in data.values[:]]
        mod2 = [np.cos(x[2])**2 + np.sin(x[2])**2 for x in data.values[:]]
        ones = np.ones(len(mod1))
        
        assert np.allclose(mod1, ones), "The first rope isn't 1"
        assert np.allclose(mod2, ones), "The second rope ins't 1"
        
    def test_len_newdata(self, new_data):
        """ the lenght in the two both pendulum must be 1 """
        mod1 = [np.cos(x[0])**2 + np.sin(x[0])**2 for x in new_data[:]]
        mod2 = [np.cos(x[2])**2 + np.sin(x[2])**2 for x in new_data[:]]
        ones = np.ones(len(mod1))
        
        assert np.allclose(mod1, ones), "The first rope isn't 1"
        assert np.allclose(mod2, ones), "The second rope ins't 1"
        
    def test_energy(self, data, new_data):
        """ the energy does not change over time """
        energy1 = methods.calculate_total_energy(pd.DataFrame(data))
        energy2 = methods.calculate_total_energy(pd.DataFrame(new_data))
        
        assert np.isclose(energy1[0], energy1[-1], atol=1e-2), "The simulation energy changes over time"
        assert np.isclose(energy2[0], energy2[-1], atol = 1e-2), "The prediction energy changes over time"
        
    def test_decreasing_loss(self, history):
        """ The loss function is decreasing over the epochs """
        assert history.history['loss'][-1] < history.history['loss'][0], "The history is not decreasing"
    
"""    
if __name__ == "__main__":
    df = prediction.data
    df_final = prediction.new_data
    
    test = Test_Prediction(df, df_final)
    
    test.test_len_df()
    test.test_len_newdata()
"""    