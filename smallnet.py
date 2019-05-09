"""
This is the small architecture used for the MNIST experiments.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model


def create_model(input_tensor, 
             input_shape,
             num_classes,
             pretrained=False):
    
    input_tensor = Input(tensor = input_tensor,
                        shape = input_shape)
    x = Conv2D(32,3,activation='relu',padding='same')(input_tensor)
    x = MaxPool2D(2,strides=2)(x)
    
    x = Conv2D(64,3,activation='relu',padding='same')(x)
    x = MaxPool2D(2,strides=2)(x)    
    
    x = Conv2D(128,3,activation='relu',padding='same')(x)
    x = MaxPool2D(2,strides=2)(x)
    
    x = Flatten()(x)  
    x = Dense(128, activation='relu')(x)
    x = Dropout(.5)(x)    
    x = Dense(num_classes, activation='linear')(x)
    
    return Model(inputs = input_tensor,
                outputs = x)