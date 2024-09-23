# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 19:00:25 2024

@author: Juanjo
"""

from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np


def masked_MAPE(v, v_, axis = None):
    """ 
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :return: int, MAPE averages on all elements of input.
    """
    mask = (v == 0)
    percentage = np.abs(v_ - v)/(np.abs(v))
    if np.any(mask):
        masked_array = np.ma.masked_array(percentage, mask = mask) # mask the dividing-zero as invalid
        result = masked_array.mean(axis = axis)
        if isinstance(result, np.ma.MaskedArray):
            return result.filled(np.nan)
        else:
            return result
    return np.mean(percentage, axis).astype(np.float64)

def MAPE(v, v_, axis = None):
    """
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :param axis: axis to do calculation.
    :returns: int, MAPE averages on all elements of input.
    """
    
    mape = (np.abs(v_ - v)/ np.abs(v) + 1e-5).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape, axis)

def RMSE(v, v_, axis = None):
    """
    Parameters
    ----------
    Mean squared error
    v : TYPE np.ndarray or int
        DESCRIPTION ground truth
    v_ : TYPE np.ndarray or int
        DESCRIPTION. prediction
    axis : TYPE, optional
        DESCRIPTION. The default is None. axis to do the calculation

    Returns int, RMSE averages on all elements of inputs
    -------
    None.

    """
    return np.sqrt(np.mean((v_ - v)**2, axis)).astype(np.float64)
    

def MAE(v, v_, axis = None):
    """

    Parameters
    ----------
    Mean absolute error
    v : TYPE np.ndarray or int
        DESCRIPTION ground truth
    v_ : TYPE np.ndarray or int
        DESCRIPTION. prediction
    axis : TYPE, optional
        DESCRIPTION. The default is None. axis to do the calculation

    Returns int, MAE averages on all elements of input
    -------
    None.

    """
    return np.mean(np.abs(v_-v), axis).astype(np.float64)

def evaluate(y, y_hat, by_step = False, by_node = False):
    """

    Parameters
    ----------
    y : TYPE array in shape of [count, time_step, node]
        DESCRIPTION.
    y_hat : TYPE array in same shape with y.
        DESCRIPTION.
    by_step : TYPE, optional evaluate by time_step dim
        DESCRIPTION. The default is False.
    by_node : TYPE, optional evaluate by node dim
        DESCRIPTION. The default is False.

    Returns array of mape, mae and rmse
    -------
    None.

    """
    
    if not by_step and not by_node:
        return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)
    if by_step and by_node: 
        return MAPE(y, y_hat, axis = 0), MAE(y, y_hat, axis = 0), RMSE(y, y_hat)
    if by_step:
        return MAPE(y, y_hat, axis = (0, 2)), MAE(y, y_hat, axis = (0, 2)), RMSE(y, y_hat, axis=(0, 2))
    if by_node:
        return MAPE(y, y_hat, axis = (0, 1)), MAE(y, y_hat, axis = (0, 1)), RMSE(y, y_hat, axis = (0, 1))
    
def test_stationarity(y):
    """
    Parameters
    ----------
    y : TYPE np.ndarray
        DESCRIPTION. Time serie

    Returns 
    -------
    None.

    """
    dftest = adfuller(y, autolag = 't-stat')
    dfoutput = pd.Series(dftest[0:4], index = ['Test', 'p-value', 'number_of_lags', 'number_of_observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical value(%s)'%key] = value
    dfoutput.to_csv('test_DF_' + str(y[0]) + '.csv')
    print(dfoutput)