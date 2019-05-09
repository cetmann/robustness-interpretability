"""
This file defines how to handle the ImageNet dataset from tfrecord files. The tfrecord files used in this work were
created using the code from
https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_imagenet_data.py
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


this_folder = os.path.dirname(os.path.abspath(__file__)) + '/'

config = configparser.ConfigParser()
config.read(this_folder + 'dataset_paths.ini')
base_folder = config['PATHS'].get('ImageNet')

# Load a class dictionary that matches the pre-trained
# encoding.
labels_dict = {}
with open(this_folder + 'imagenet_labels.csv', 'rt') as csvfile:
    file_contents = csv.reader(csvfile, delimiter=',')
    for row in file_contents:
        labels_dict[row[0]] = row[1]

##### Training ###########
def collect_train_data(num_samples_per_class=0):
    print("Collecting training data...")
    tfrecord_folder = base_folder + 'tfrecord/'
    file_list = os.listdir(tfrecord_folder)
    train_data = [(tfrecord_folder + f) for f in file_list if 'train-' in f]
    # Create dummy labels, because the label information is contained
    # in the train_data files.
    train_labels = np.zeros_like(train_data,dtype=np.int32)
    return train_data, train_labels

def collect_val_data():
    print("Collecting validation data...")
    tfrecord_folder = base_folder + 'tfrecord/'
    file_list = os.listdir(tfrecord_folder)
    val_data = [(tfrecord_folder + f) for f in file_list if 'validation-' in f]
    # Create dummy labels, because the label information is contained
    # in the train_data files.
    val_labels = np.zeros_like(val_data,dtype=np.int32)
    return val_data, val_labels

# tf.data batch iterator for the training data
def train_BI(filenames, 
             labels, 
             batch_size,
             num_parallel_calls):
    dataset = tf.data.TFRecordDataset(filenames)
    batch_prepare = lambda image: image_preprocessing(image,
                                                      None,
                                                      file_type = 'tfrecord',
                                                      shape = [256,256],
                                                      random_events = True,
                                                      data_augmentation = True,
                                                      additive_noise = False,
                                                      subtract_by = 'ImageNet',
                                                      divide_by = 1.,
                                                      colorspace = 'BGR',
                                                      min_rescale = 258,
                                                      rescale = True,
                                                      noise_level = 10.,
                                                      clip_values = bounds())
    dataset = dataset.map(batch_prepare,num_parallel_calls=num_parallel_calls)    
    batched_dataset = dataset.batch(batch_size,
                                   drop_remainder = False)       
    train_batch_iterator = batched_dataset.make_initializable_iterator()
    return train_batch_iterator

# tf.data batch iterator for the validation data
def val_BI(filenames, 
           labels, 
           batch_size,
           num_parallel_calls):
    dataset = tf.data.TFRecordDataset(filenames)
    batch_prepare = lambda image: image_preprocessing(image, 
                                                      None,
                                                      file_type = 'tfrecord',
                                                      shape = [256,256],
                                                      random_events = False,
                                                      data_augmentation = False,
                                                      additive_noise = False,
                                                      subtract_by = 'ImageNet',
                                                      divide_by = 1.,
                                                      colorspace = 'BGR',
                                                      min_rescale = 258,
                                                      rescale = True)
    dataset = dataset.map(batch_prepare,
                          num_parallel_calls=num_parallel_calls)
    batched_dataset = dataset.batch(batch_size,
                                   drop_remainder = False)       
    batch_iterator = batched_dataset.make_initializable_iterator()
    return batch_iterator

# Additional tf.data batch iterator for the data that is used just for the propagation
# of a few images for visualization.
def img_BI(filenames, 
           labels, 
           batch_size,
           num_parallel_calls):
    dataset = tf.data.TFRecordDataset(filenames)
    batch_prepare = lambda image: image_preprocessing(image, 
                                                      None,
                                                      file_type = 'tfrecord',
                                                      shape = [256,256],
                                                      random_events = False,
                                                      data_augmentation = False,
                                                      additive_noise = False,
                                                      subtract_by = 'ImageNet',
                                                      divide_by = 1.,
                                                      colorspace = 'BGR',
                                                      min_rescale = 258,
                                                      rescale = True)
    dataset = dataset.map(batch_prepare,
                          num_parallel_calls=num_parallel_calls)
    batched_dataset = dataset.batch(batch_size,
                                   drop_remainder = False)       
    batch_iterator = batched_dataset.make_initializable_iterator()
    return batch_iterator

def interpret_as_image(image):
    return preprocessing.interpret_as_image(image,
                                           add_by='ImageNet',
                                           colorspace='BGR')

def num_train_samples():
    return 1281167

def num_val_samples():
    return 50000

def bounds():
    # This is a little problematic here. Foolbox only allows
    # for scalar bounds, not bounds per channel. For this reason,
    # we use the worst-case bounds here.
    return (-130., 255.-100.)

min_values = np.array([0.,0.,0.],np.float32)
max_values = np.array([1.,1.,1.],np.float32)
def image_range():
    return [0.,255.]