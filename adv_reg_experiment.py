"""
All models are trained using this base script. It can be run via
python adv_reg_experiment.py experiment.ini
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import configparser
import argparse
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import utils
import dataset
import robust_model
import training
import summary_utils


parser = argparse.ArgumentParser(
    description='Train an adversarially robust network.')
parser.add_argument('ini_file', type=str)
args = parser.parse_args()

ini_file = args.ini_file


# Load the desired specifications from the provided
# INI-file.
config = configparser.ConfigParser()
config.read(ini_file)
paths = config['PATHS']
hyperparameters = config['HYPERPARAMETERS']
logging = config['LOGGING']
architecture = config['ARCHITECTURE']

# PATHS
base_name = os.path.splitext(
    os.path.basename(ini_file))[0]

tensorboard_logdir = paths.get(
    'tensorboard_logdir') + \
    base_name + '/'
dataset_name = paths.get(
    'dataset_name')
saved_model_folder = paths.get(
    'saved_model_folder')
saved_model_path = saved_model_folder + \
    base_name + '/model.ckpt'

# HYPERPARAMETERS
lr_decrease_interval = int(hyperparameters.get(
    'lr_decrease_interval'))
lr_decrease_factor = float(hyperparameters.get(
    'lr_decrease_factor'))
batch_size_per_gpu = int(hyperparameters.get(
    'batch_size_per_gpu'))
robust_regularization = hyperparameters.getboolean(
    'robust_regularization',
    True)
use_wavelet_decomposition = hyperparameters.getboolean(
    'use_wavelet_decomposition',
    True)
sensitivity_mode = hyperparameters.get(
    'sensitivity_mode',
    'NLL')

wavelet_weights = [float(i) 
                   for i in eval(
                       hyperparameters.get(
                           'wavelet_weights',
                           [4,2,1,0]
                       )
                   )
                  ]
decomp_type = hyperparameters.get(
    'decomp_type')
lp_wavelet_parameter= float(hyperparameters.get(
    'lp_wavelet_parameter'))
p_norm = float(hyperparameters.get(
    'p_norm'))
learning_rate_at_start= float(hyperparameters.get(
    'learning_rate_at_start'))
weight_decay_parameter = float(hyperparameters.get(
    'weight_decay_parameter'))
bn_momentum_value = float(hyperparameters.get(
    'bn_momentum_value'))
num_epochs = int(hyperparameters.get(
    'num_epochs'))
learning_phase = int(hyperparameters.get(
    'learning_phase',
    1))

# LOGGING
train_summary_period = int(logging.get(
    'train_summary_period'))
val_summary_period = int(logging.get(
    'val_summary_period'))
adversarial_test_period = int(logging.get(
    'adversarial_test_period'))
num_adversarial_batches = int(logging.get(
    'num_adversarial_batches'))

# ARCHITECTURE
Model = architecture['model']
pretrained = architecture.getboolean(
    'pretrained',
    False)

tf.logging.set_verbosity(tf.logging.INFO)

# Create the saved_model_folder if it does not
# exist yet. Otherwise, tf.Saver throws an error.
pathlib.Path(saved_model_folder).mkdir(parents=True,
                                       exist_ok=True) 
summary_writer = tf.summary.FileWriter(
    tensorboard_logdir,
    None,
    flush_secs=30)


d_y, d_x = dataset.image_resolution[dataset_name]
num_classes = dataset.num_classes[dataset_name]
num_parallel_calls = np.int32(512)


# Create the necessary placeholders, which represent
# the inputs and hyperparameters of the network.
if dataset_name == 'MNIST':
    files = tf.placeholder(tf.float32,name='files')
else:
    files = tf.placeholder(tf.string,name='files')
labels = tf.placeholder(tf.int32,name='labels')

if dataset_name == 'MNIST':
    x = tf.placeholder(tf.float32,[None,d_y,d_x,1],name='x')
    d_c = 1
else:
    x = tf.placeholder(tf.float32,[None,d_y,d_x,3],name='x')
    d_c = 3
l1_p = tf.placeholder(tf.float32,(),name='l1_parameter')
l2_p = tf.placeholder(tf.float32,(),name='l2_parameter')
lp_wavelet_p = tf.placeholder(tf.float32,(),name='lp_wavelet_parameter')
weight_decay_p = tf.placeholder(tf.float32,(),name='weight_decay_parameter')
starter_learning_rate = tf.placeholder(tf.float32,(),name='starter_learning_rate')
bn_momentum = tf.placeholder(tf.float32,(),name='bn_momentum')
batch_size = tf.placeholder(tf.int64,(),name='batch_size')                            
epoch_step = tf.Variable(0, trainable=False)                        
batch_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, 
                                       epoch_step,
                                       lr_decrease_interval,
                                       lr_decrease_factor,
                                       staircase=True)

config=tf.ConfigProto(
    allow_soft_placement = True,
    log_device_placement = True)
graph = tf.get_default_graph()
session = tf.Session(graph=graph,
                     config=config)
K.set_session(session)
K.set_image_data_format('channels_last')
session.run(K.learning_phase(), feed_dict={K.learning_phase(): 0})
session.as_default()


# Depending on the model specified in the INI-file,
# load the preferred model. The create_model method
# has to output a tf.keras.models.Model object. If
# the model has BatchNorm layers, the output of this
# model should be a list where the the first entry is
# the logit output of the network and the remaining entries
# should be the Keras tensors which are the incoming nodes
# to these BatchNorm layers. When using BatchNorm with
# the proposed robust regularization, the used_fused
# parameter needs to be 'False'. If no BatchNorm is used,
# the output of the model should just be the logit output
# of the network.
if Model == 'VGG16':
    from vgg16 import create_model
if Model == 'ResNet6':
    from resnet6 import create_model
if Model == 'ResNet18':
    from resnet18 import create_model
if Model == 'ResNet50':
    from resnet50 import create_model
if Model == 'SmallNet':
    from smallnet import create_model
model = create_model(input_tensor=x,
              input_shape = [d_y,d_x,d_c],
              num_classes = num_classes,
              pretrained = pretrained)


# Create the optimizer.
optimizer = tf.train.MomentumOptimizer(
    learning_rate=learning_rate,
    momentum=.9)

# Load the dataset object which automatically
# handles the data loading and preprocessing
# and create iterator objects.
dataset = dataset.dataset(
    files_ph = files,
    labels_ph = labels,
    batch_size_ph = batch_size,
    dataset_name = dataset_name)


# We use a string training handle to allow for quick switching
# between the different batch iterators.
handle = tf.placeholder(tf.string)
iterator = tf.data.Iterator.from_string_handle(
    handle,
    dataset.train_batch_iterator.output_types,
    dataset.train_batch_iterator.output_shapes)


# Create a robust model according to our specifications.
robust_model = robust_model.robust_model(iterator = iterator,
                                        session = session,
                                        model = model,
                                        num_classes = num_classes,
                                        optimizer = optimizer,
                                        dataset = dataset,
                                        p_norm = p_norm,
                                        decomp_type = decomp_type,
                                        learning_rate = learning_rate,
                                        weight_decay_p = weight_decay_p,
                                        lp_wavelet_p = lp_wavelet_p,
                                        batch_size = batch_size,
                                        bn_momentum = bn_momentum,
                                        robust_regularization = robust_regularization,
                                        use_wavelet_decomposition = use_wavelet_decomposition,
                                        wavelet_weights = wavelet_weights,
                                        sensitivity_mode = sensitivity_mode)
GPU_collections = robust_model.GPU_collections


# Apply these attacks every now and then in order to get an impression
# of the adversarial robustness during training.
attack_types = ['GradientAttack']
   
# Create the training and logging loop.
training_procedure = training.training(
    handle = handle,
    dataset = dataset,
    batch_size_placeholder = batch_size,
    train_op = robust_model.train_op,
    session = session,
    pretrained = pretrained,
    epoch_step = epoch_step,
    batch_step = batch_step,
    summary_writer = summary_writer,
    train_summary_op = robust_model.summary_op,
    img_summary_op = robust_model.img_summary_op,
    optimizer = optimizer,
    GPU_collections = robust_model.GPU_collections,
    adversarial_model = robust_model.adversarial_model,
    adversarial_attacks = attack_types,
    saver_path = saved_model_path,
    num_adversarial_batches = num_adversarial_batches,
    batch_size = batch_size_per_gpu,
    num_epochs = num_epochs,
    train_summary_period = train_summary_period,
    val_summary_period = val_summary_period,
    adv_summary_period = adversarial_test_period)

train_feed_dict = {starter_learning_rate: learning_rate_at_start,
                   K.learning_phase(): learning_phase,
                   weight_decay_p: weight_decay_parameter,
                   batch_size: batch_size_per_gpu,
                   lp_wavelet_p: lp_wavelet_parameter,
                   bn_momentum: bn_momentum_value}
val_feed_dict = {K.learning_phase(): 0,
                 batch_size: batch_size_per_gpu}
training_procedure.train(train_feed_dict,
                        val_feed_dict)