# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:57:11 2024

@author: Juanjo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN, Dropout

class RNNmodel(tf.keras.Model):
    def __init__(self, input_shape, units, output_size):
        super(RNNmodel, self).__init__()
        
        # define the layers of our model
        self.rnn_layer = SimpleRNN(units, return_sequences = False, input_shape = input_shape)
        self.dropout_layer = Dropout(0.2)
        self.output_layer = Dense(output_size)
        
    def call(self, inputs):
        # aply all the layers
        rnn_output = self.rnn_layer(inputs)
        # aply the dropout layer
        dropout_output = self.dropout_layer(rnn_output)
        # aply the last layer
        output = self.output_layer(dropout_output)
        
        return output
     