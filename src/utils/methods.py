# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:35:50 2024

@author: Juanjo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras as keras
from statsmodels.tsa.vector_ar.var_model import VAR


def split_set(df, t, size):
    
    """ 
    Splits the time series in two halfs
    df: numpy array, pandas dataframe: the time series
    t: nunmpy array: time
    
    """
    df_train, df_test, t_train, t_test = train_test_split(df, t, test_size = size, shuffle = False)
    
    return df_train, df_test, t_train, t_test

def scale_data(df):
    """

    Parameters
    ----------
    df : dataframe

    Returns
    -------
    df_scaled : the dataframe scaled with min in 0 and max in 1.

    """
    scaler = MinMaxScaler(feature_range = (0, 1))
    df_scaled = scaler.fit_transform(df)
    
    return df_scaled

def ANN(model, data):
    """
    Parameters
    ----------
    model : keras model
        The ANN model which we predict the other half of the data
    data : numpy array
        The first half of the data with who's we are going to predict the second half 

    Returns the second half of the data
    -------
    None.

    """
    
    # Load the model
    ANN_model = keras.models.load_model(model)
    
    # Forecast
    y_pred = ANN_model.predict(data)
    
    return y_pred

def var_multi(data, data_test):
    """
    Parameters
    ----------
    data : numpy array
        The first half of the time sieries we want to predict

    data_test : numpy array
        The second half of the time series
    Returns the second half of the time series
    -------
    None.
    """
    
    # Define the model
    model = VAR(endog = data, exog = data)
    model_fit = model.fit()
    
    # Predict the second half
    y_pred = model_fit.forecast(data, exog_future = data_test, steps = len(data_test))
    
    return y_pred

def create_sequences(data, n_steps):
    """
    Parameters
    ----------
    data : numpy array
        the time series we want to use as training dataset.
    n_steps : int
        number of points that has each sequence.

    Returns x: the sequences of n_steps points of the time series
    y: the target of each sequences (the point number n_steps + 1)
    -------
    None.
    (if this function is a hustle, you can use timeseries_dataset_from_array())
    """
    
    x, y = [], []
    for i in range(len(data) - n_steps):
        x.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(x), np.array(y)