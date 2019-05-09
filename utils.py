"""
Some utility functions. Most are not used in the paper, however.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.python.training import moving_averages

# The following function is inspired by the Tensorflow Multi-GPU
# implementation example:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

def average_gradients(GPU_grads):
    resulting_averages = []
    for grad_vars in zip(*GPU_grads):
        gradients = []
        for gradient, _ in grad_vars:
            GPUs_and_gradients = tf.expand_dims(
                gradient, 0)
            gradients.append(GPUs_and_gradients)

        gradients = tf.concat(
            axis=0,
            values=gradients)
        gradients = tf.reduce_mean(
            gradients, 
            0)
        variables = grad_vars[0][1]
        gradients_and_variables = (
            gradients,
            variables)
        resulting_averages.append(gradients_and_variables)
    return resulting_averages


def edge_filter(filter_type="simple",
                edges_per_channel=False,
                greyscale=False):
    '''
    Helper function if we want to calculate the T(G)V
    of the images in a batch.
    '''
    if filter_type == "simple": 
        # Only needs to be square because of the 'valid' convolution
        # One-sided difference operator
        edge_filter_x_atom = np.array([[1,-1],
                                       [0,0]], 
                                      dtype=np.float32) 
    if filter_type == "sobel":
        # Sobel-Operator
        edge_filter_x_atom = np.array([[1,0,-1],
                                       [2,0,-2],
                                       [1,0,-1]], 
                                      dtype=np.float32) 
    if filter_type == "scharr":
        # Scharr-Operator
        edge_filter_x_atom = np.array([[3, 0,-3],
                                       [10,0,-10],
                                       [3, 0,-3]],
                                      dtype=np.float32) 
    if filter_type == "laplace":
        # Laplace-Operator
        edge_filter_x_atom = np.array([[0,-1,0],
                                       [-1,4,-1],
                                       [0,-1,0]], 
                                      dtype=np.float32) 
    edge_filter_y_atom = edge_filter_x_atom.T
        
    filter_size_x = edge_filter_x_atom.shape[1]
    filter_size_y = edge_filter_x_atom.shape[0]

    # Right format: NHWC images => HWIO filters
    if edges_per_channel:
        # Version 1: Edges on each color channel
        edge_filter_x = np.zeros((filter_size_y,filter_size_x,3,3), 
                                 dtype=np.float32)
        edge_filter_x[:,:,0,0] = edge_filter_x_atom
        edge_filter_x[:,:,1,1] = edge_filter_x_atom
        edge_filter_x[:,:,2,2] = edge_filter_x_atom
            
        edge_filter_y = np.zeros((filter_size_x,filter_size_y,3,3), 
                                 dtype=np.float32)
        edge_filter_y[:,:,0,0] = edge_filter_y_atom
        edge_filter_y[:,:,1,1] = edge_filter_y_atom
        edge_filter_y[:,:,2,2] = edge_filter_y_atom
    
    else:
        # Version 2: Edges on grayscale images
        # Might lead to differently colored edges equalizing
        edge_filter_x = np.zeros((filter_size_y,filter_size_x,3,1), 
                                 dtype=np.float32)
        edge_filter_x[:,:,0,0] = edge_filter_x_atom
        edge_filter_x[:,:,1,0] = edge_filter_x_atom
        edge_filter_x[:,:,2,0] = edge_filter_x_atom
            
        edge_filter_y = np.zeros((filter_size_x,filter_size_y,3,1), 
                                 dtype=np.float32)
        edge_filter_y[:,:,0,0] = edge_filter_y_atom
        edge_filter_y[:,:,1,0] = edge_filter_y_atom
        edge_filter_y[:,:,2,0] = edge_filter_y_atom
    if greyscale:
        # Version 2: Edges on grayscale images
        # Might lead to differently colored edges equalizing
        edge_filter_x = np.zeros((filter_size_y,filter_size_x,1,1), 
                                 dtype=np.float32)
        edge_filter_x[:,:,0,0] = edge_filter_x_atom
            
        edge_filter_y = np.zeros((filter_size_x,filter_size_y,1,1), 
                                 dtype=np.float32)
        edge_filter_y[:,:,0,0] = edge_filter_y_atom

    edge_filter_x = tf.constant(edge_filter_x, 
                                dtype=tf.float32)
    edge_filter_y = tf.constant(edge_filter_y, 
                                dtype=tf.float32)
    
    return edge_filter_x, edge_filter_y

def isotropic_TV(tensor,
                 normalize=True,
                 filter_type="simple",
                 edges_per_channel=False,
                 eps=1e-6):
    '''
    Isotropic mean total variation of a batch.
    '''
    edge_filter_x, edge_filter_y = edge_filter(filter_type,
                                               edges_per_channel)
    if tensor.shape[-1] == 1:
        edge_filter_x, edge_filter_y = edge_filter(filter_type,
                                                   greyscale=True)
    if normalize:
        tensor_norm = tf.sqrt(tf.reduce_sum(
            tensor**2,axis=[1,2,3],keepdims=True) + eps )
        tensor = tensor / tensor_norm
    edges_x_of_grads = tf.nn.conv2d((tensor), 
                                    edge_filter_x, 
                                    strides=[1,1,1,1], 
                                    padding='VALID')            
    edges_y_of_grads = tf.nn.conv2d((tensor), 
                                    edge_filter_y, 
                                    strides=[1,1,1,1], 
                                    padding='VALID')
    edge_image = tf.sqrt(edges_x_of_grads**2 + edges_y_of_grads**2 + eps)
    isotropic_TV = tf.reduce_sum(edge_image,axis=[1,2,3])
    iso_TV = tf.reduce_mean(isotropic_TV, name='iso_TV')
    tf.add_to_collection('TV_losses', iso_TV)
    return iso_TV

def anisotropic_TV(tensor,
                   normalize=True,
                   filter_type="simple",
                   edges_per_channel=False,
                   eps=1e-6):
    '''
    Anisotropic mean total variation of a batch.
    '''
    edge_filter_x, edge_filter_y = edge_filter(filter_type,
                                               edges_per_channel)
    if tensor.shape[-1] == 1:
        edge_filter_x, edge_filter_y = edge_filter(filter_type,
                                                   greyscale=True)
    if normalize:
        tensor_norm = tf.sqrt(tf.reduce_sum(
            tensor**2,axis=[1,2,3],keepdims=True) + eps )
        tensor = tensor / tensor_norm
    edges_x_of_grads = tf.nn.conv2d((tensor), 
                                    edge_filter_x, 
                                    strides=[1,1,1,1], 
                                    padding='VALID')            
    edges_y_of_grads = tf.nn.conv2d((tensor), 
                                    edge_filter_y, 
                                    strides=[1,1,1,1], 
                                    padding='VALID')
    edge_image = tf.abs(edges_x_of_grads) + tf.abs(edges_y_of_grads)
    anisotropic_TV = tf.reduce_sum(edge_image,axis=[1,2,3])
    aniso_TV = tf.reduce_mean(anisotropic_TV, name='aniso_TV')
    tf.add_to_collection('TV_losses', aniso_TV)
    return aniso_TV


def lp_norm(tensor,
            p_norm=2,
            normalize=False,
            eps=1e-6):
    '''
    The p-norm of a tensor. Can be l2-normalized
    optionally.
    '''
    if tensor is list:
        a_1 = lp_norm(tensor[0], p_norm = p_norm)
        d_1 = lp_norm(tensor[1], p_norm = p_norm)
        d_2 = lp_norm(tensor[2], p_norm = p_norm)
        d_3 = lp_norm(tensor[3], p_norm = p_norm)
    else:
        if normalize:
            tensor_norm = tf.sqrt(tf.reduce_sum(
                tensor**2,axis=[1,2,3],keepdims=True) + eps )
            tensor = tensor / tensor_norm    
        norm = tf.reduce_sum(tf.abs(tensor)**p_norm,
                            axis=[1,2,3],
                            name='regularization_lp')
        avg_norm = tf.reduce_mean(norm)
    return avg_norm

def lp_norm_weighted(tensor,
                     weights,
                     p_norm = 2):
    '''
    A nested weight list and a wavelet decomposition
    are combined to form a weighted p-norm of
    the wavelet decomposition.
    '''
    if type(tensor) == list:
        [c_0,c_1,c_2,c_3] = weights
        
        
        # c_0 and a_0 are nested lists
        a_0 = lp_norm_weighted(tensor[0], c_0, p_norm)
        d_1 = lp_norm_weighted(tensor[1], c_1, p_norm)
        d_2 = lp_norm_weighted(tensor[2], c_2, p_norm)
        d_3 = lp_norm_weighted(tensor[3], c_3, p_norm)

        return a_0 + d_1 + d_2 + d_3
    else:
        norm = tf.reduce_sum(tf.abs(tensor)**p_norm,
                            axis=[1,2,3],
                            name='regularization_lp')
        avg_norm = tf.reduce_mean(norm)
        if type(weights) != list:
            avg_norm*= weights
    return avg_norm

def nested_weight_list(weight_list):
    '''
    This function creates a nested list of weights that
    follows the same list structure as the wavelet
    decomposition.
    '''
    if len(weight_list)>1:
        w = weight_list[0]
        return [nested_weight_list(weight_list[1:]),
                w,w,w]
    else:
        return weight_list[0]    


def add_bn_ops(model,bn_inputs,bn_momentum=.999):
    '''
    If we propagate a tf.tensor through the keras
    model for different GPUs, the update operators
    do not get updated correctly. This is why we 
    create the correct update operators ourselves
    manually here.
    '''
    for l in model.layers:
        # Check for BatchNorm layers
        if hasattr(l,'gamma'):
            tf.add_to_collection('bn_gammas', 
                                 l.gamma)
            tf.add_to_collection('bn_betas', 
                                 l.beta)
            tf.add_to_collection('bn_means', 
                                    l.moving_mean)
            tf.add_to_collection('bn_vars', 
                                 l.moving_variance)
            tf.add_to_collection('bn_initializers',
                                 l.moving_mean.initializer)
            tf.add_to_collection('bn_initializers',
                                 l.moving_variance.initializer)
    bn_update_ops = []
    bn_means = tf.get_collection('bn_means')
    bn_vars = tf.get_collection('bn_vars')
    for act, m, v in zip(bn_inputs, bn_means, bn_vars): 
        input_shape = K.int_shape(act)
        reduction_axes = list(range(len(input_shape))) 
        del reduction_axes[-1]
        mean, variance = tf.nn.moments(act, reduction_axes)
        m_mean_op = moving_averages.assign_moving_average(m,
                                                          mean,
                                                          bn_momentum,
                                                          zero_debias = False)
        tf.add_to_collection('bn_update_ops',m_mean_op)
        m_var_op = moving_averages.assign_moving_average(v,
                                                         variance,
                                                         bn_momentum,
                                                         zero_debias = False)
        tf.add_to_collection('bn_update_ops',m_var_op)
        
def select(tensor, indices, max_value):
    sparse_tensor = tensor*tf.one_hot(indices, max_value)
    selected_values = tf.reduce_sum(sparse_tensor,
                                    axis = 1)
    return selected_values