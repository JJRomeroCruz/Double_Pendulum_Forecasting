"""
Created on Tue Sep 17 10:57:11 2024

@author: Juanjo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN, Dropout

class RNNmodel(tf.keras.Model):
    def __init__(self, input_shape, units, output_size, *args, **kwargs):
        super(RNNmodel, self).__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.units = units
        self.output_size = output_size
        
        # define the layers of our model
        self.rnn_layer1 = SimpleRNN(self.units, return_sequences = True, input_shape = self.input_shape)
        self.dropout_layer1 = Dropout(0.2)
        self.rnn_layer2 = SimpleRNN(self.units, return_sequences = True, input_shape = self.input_shape)
        self.dropout_layer2 = Dropout(0.2)
        self.rnn_layer3 = SimpleRNN(self.units, return_sequences = False, input_shape = self.input_shape)
        self.dropout_layer3 = Dropout(0.2)
        self.output_layer = Dense(self.output_size)
        
    def call(self, inputs):
        # aply all the layers
        rnn_output = self.rnn_layer1(inputs)
        # aply the dropout layer
        dropout_output = self.dropout_layer1(rnn_output)
        # rnn layer
        x = self.rnn_layer2(dropout_output)
        # dropout
        x = self.dropout_layer2(x)
        # rnn layer
        x = self.rnn_layer3(x)
        # dropout layer
        x = self.dropout_layer3(x)
        # aply the last layer
        output = self.output_layer(x)
        
        return output
        
    def get_config(self):
        base_config = super().get_config()
        config = {
            'input_shape': keras.saving.serialize_keras_object(self.input_shape),
            'units': keras.saving.serialize_keras_object(self.units), 
            'output_size': keras.saving.serialize_keras_object(self.output_size)
            }
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config):
        input_shape_config = config.pop('input_shape')
        input_shape = keras.saving.deserialize_keras_object(input_shape_config)
        
        units_config = config.pop('units')
        units = keras.saving.deserialize_keras_object(units_config)
        
        output_size_config = config.pop('output_size')
        output_size =  keras.saving.deserialize_keras_object(output_size_config)
        
        return cls(input_shape, units, output_size, **config)
   