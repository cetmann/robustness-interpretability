"""
This file defines how to handle the MNIST dataset.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import csv
import preprocessing
from preprocessing import image_preprocessing
import configparser
import gzip
import pickle

this_folder = os.path.dirname(os.path.abspath(__file__)) + '/'

config = configparser.ConfigParser()
config.read(this_folder + 'dataset_paths.ini')
base_folder = config['PATHS'].get('MNIST')




##### Training ###########
def collect_train_data(num_samples_per_class=0):
    f = gzip.open(base_folder + 'mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()

    train_data = np.array(train_set[0], dtype='float32').reshape((-1,28,28,1))
    train_labels = np.array(train_set[1], dtype='int32')
    return train_data, train_labels

def collect_val_data():
    print("Collecting validation data...")
    f = gzip.open(base_folder + 'mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    val_data = np.array(valid_set[0], dtype='float32').reshape((-1,28,28,1))
    val_labels = np.array(valid_set[1], dtype='int32')
    return val_data, val_labels

# tf.data batch iterator for the training data
def train_BI(images, 
             labels, 
             batch_size,
             num_parallel_calls=100):
    dataset = tf.data.Dataset.from_tensor_slices(
        (images, labels))  
    batched_dataset = dataset.batch(batch_size,
                                   drop_remainder = False)       
    train_batch_iterator = batched_dataset.make_initializable_iterator()
    return train_batch_iterator

# tf.data batch iterator for the validation data
def val_BI(images, 
           labels, 
           batch_size,
           num_parallel_calls=100):
    dataset = tf.data.Dataset.from_tensor_slices(
        (images, labels))  
    batched_dataset = dataset.batch(batch_size,
                                   drop_remainder = False)       
    val_batch_iterator = batched_dataset.make_initializable_iterator()
    return val_batch_iterator

# Additional tf.data batch iterator for the data that is used just for the propagation
# of a few images for visualization.
def img_BI(images, 
           labels, 
           batch_size,
           num_parallel_calls=100):
    dataset = tf.data.Dataset.from_tensor_slices(
        (images, labels))  
    batched_dataset = dataset.batch(batch_size,
                                   drop_remainder = False)       
    img_batch_iterator = batched_dataset.make_initializable_iterator()
    return img_batch_iterator

def interpret_as_image(image):
    return image

def num_train_samples():
    return 50000

def num_val_samples():
    return 1000

def bounds():
    # This is a little problematic here. Foolbox only allows
    # for scalar bounds, not bounds per channel. For this reason,
    # we use the worst-case bounds here.
    return (0,1)

min_values = np.array([0.,0.,0.],np.float32)
max_values = np.array([1.,1.,1.],np.float32)
def image_range():
    return [0.,1.]