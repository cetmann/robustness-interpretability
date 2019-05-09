"""ResNet50 model for Keras.
This is an adapted version of the official ResNet50 
implementation for Keras, which in turn was was adapted
from a contribution by 'BigMoyan'.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

#from keras import get_submodules_from_kwargs

import tensorflow
#import keras
import keras_applications
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Activation,
    GlobalAveragePooling2D, Dense, add, ZeroPadding2D, Input, MaxPooling2D)
from keras_applications import imagenet_utils
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model


this_folder = os.path.dirname(os.path.abspath(__file__)) + '/'
preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')

use_fused = False
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    bn_inputs = []
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1,
               (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2a')(input_tensor)
    bn_inputs.append(x)
    x = BatchNormalization(axis=bn_axis,
                           name=bn_name_base + '2a',
                           fused=use_fused)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2,
                      kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    bn_inputs.append(x)
    x = BatchNormalization(axis=bn_axis,
                           name=bn_name_base + '2b',
                           fused=use_fused)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3,
               (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c')(x)
    bn_inputs.append(x)
    x = BatchNormalization(axis=bn_axis,
                           name=bn_name_base + '2c',
                           fused=use_fused)(x)
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x, bn_inputs


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    bn_inputs = []
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, 
               (1, 1), 
               strides=strides,
               kernel_initializer='he_normal',
               name=conv_name_base + '2a')(input_tensor)
    bn_inputs.append(x)
    x = BatchNormalization(axis=bn_axis,
                           name=bn_name_base + '2a',
                           fused=use_fused)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, 
               kernel_size,
               padding='same',
               kernel_initializer='he_normal',
               name=conv_name_base + '2b')(x)
    bn_inputs.append(x)
    x = BatchNormalization(axis=bn_axis,
                           name=bn_name_base + '2b',
                           fused=use_fused)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3,
               (1, 1),
               kernel_initializer='he_normal',
               name=conv_name_base + '2c')(x)
    bn_inputs.append(x)
    x = BatchNormalization(axis=bn_axis,
                           name=bn_name_base + '2c',
                           fused=use_fused)(x)

    shortcut = Conv2D(filters3,
                      (1, 1),
                      strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '1')(input_tensor)
    bn_inputs.append(shortcut)
    shortcut = BatchNormalization(axis=bn_axis,
                                  name=bn_name_base + '1',
                                  fused=use_fused)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x, bn_inputs


def create_model(input_tensor, 
             input_shape,
             num_classes,
             pretrained = False,
             **kwargs):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    keras_utils = keras_applications._KERAS_UTILS



    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    bn_inputs = []
    img_input = Input(tensor=input_tensor, shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    bn_inputs.append(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1', fused=use_fused)(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x, bn_inputs_ = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    bn_inputs = bn_inputs + bn_inputs_
    x, bn_inputs_ = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    bn_inputs = bn_inputs + bn_inputs_
    x, bn_inputs_ = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    bn_inputs = bn_inputs + bn_inputs_

    x, bn_inputs_ = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    bn_inputs = bn_inputs + bn_inputs_
    x, bn_inputs_ = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    bn_inputs = bn_inputs + bn_inputs_
    x, bn_inputs_ = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    bn_inputs = bn_inputs + bn_inputs_
    x, bn_inputs_ = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    bn_inputs = bn_inputs + bn_inputs_

    x, bn_inputs_ = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    bn_inputs = bn_inputs + bn_inputs_
    x, bn_inputs_ = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    bn_inputs = bn_inputs + bn_inputs_
    x, bn_inputs_ = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    bn_inputs = bn_inputs + bn_inputs_
    x, bn_inputs_ = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    bn_inputs = bn_inputs + bn_inputs_
    x, bn_inputs_ = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    bn_inputs = bn_inputs + bn_inputs_
    x, bn_inputs_ = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    bn_inputs = bn_inputs + bn_inputs_

    x, bn_inputs_ = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    bn_inputs = bn_inputs + bn_inputs_
    x, bn_inputs_ = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    bn_inputs = bn_inputs + bn_inputs_
    x, bn_inputs_ = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    bn_inputs = bn_inputs + bn_inputs_

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(num_classes,
              activation='linear',
              name='fc1000')(x)

    # Create model.
    model = Model(img_input, [x] + bn_inputs, name='resnet50')

    # Load weights.
    if pretrained == True:
        print("Loading pretrained model..")
        weights_path = tensorflow.keras.utils.get_file(
            'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
            WEIGHTS_PATH,
            cache_subdir='models',
            md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        model.load_weights(weights_path)

    return model
