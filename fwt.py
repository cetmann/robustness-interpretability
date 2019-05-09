"""
This file defines the fast wavelet transform, which did not end up being used in the paper.
"""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pywt as pywt

def create_filter_bank(wavelet_type='bior2.2'):
    """ 
    This functions implements a 2D filter bank.
    wavelet_type -- str. A pywt-compatible wavelet type.
    """
    
    # Load the wavelet from pywt
    w = pywt.Wavelet(wavelet_type)
    
    # Tensorflow implements convolutional operators
    # as cross-correlations, which is equivalent to
    # convolutions with a flipped kernel. Flip the
    # wavelets in order to be compatible with
    # Tensorflow-style convolution.
    dec_hi = w.dec_hi[::-1]
    dec_lo = w.dec_lo[::-1]
    rec_hi = w.rec_hi
    rec_lo = w.rec_lo

    # The filter banks need to be at least of size 4
    # in order to work with the below-implemented
    # padding. Fill up with zeros.
    for l in [dec_hi, dec_lo, rec_hi, rec_lo]:
        while len(l)<4:
            l = [0.] + l

    # Turn the lists of numbers into Tensorflow constants
    dec_hi = tf.constant(dec_hi) 
    dec_lo = tf.constant(dec_lo)
    rec_hi = tf.constant(rec_hi)
    rec_lo = tf.constant(rec_lo)

    # Separable 2D scaling functions and wavelets are realized through 
    # tensor products of the 1D scaling functions and wavelets.
    lo_lo_dec = tf.expand_dims(dec_lo,0)*tf.expand_dims(dec_lo,1)
    lo_hi_dec = tf.expand_dims(dec_lo,0)*tf.expand_dims(dec_hi,1)
    hi_lo_dec = tf.expand_dims(dec_hi,0)*tf.expand_dims(dec_lo,1)
    hi_hi_dec = tf.expand_dims(dec_hi,0)*tf.expand_dims(dec_hi,1)
    
    lo_lo_rec = tf.expand_dims(rec_lo,0)*tf.expand_dims(rec_lo,1)
    lo_hi_rec = tf.expand_dims(rec_lo,0)*tf.expand_dims(rec_hi,1)
    hi_lo_rec = tf.expand_dims(rec_hi,0)*tf.expand_dims(rec_lo,1)
    hi_hi_rec = tf.expand_dims(rec_hi,0)*tf.expand_dims(rec_hi,1)

    # Turn this filter bank into a Tensorflow-compatible
    # convolutional kernel. The convention is 
    # conv2d shape = [H,W,I,O] and
    # conv2d_transpose shape = [H,W,O,I] and
    # This means that
    # filter_bank_dec.shape = [H,W,1,4] and also
    # filter_bank_rec.shape = [H,W,1,4]
    
    filter_bank_dec = tf.stack([lo_lo_dec,lo_hi_dec,hi_lo_dec,hi_hi_dec],
                           axis=2)
    filter_bank_dec = tf.expand_dims(filter_bank_dec,2)
    
    filter_bank_rec = tf.stack([lo_lo_rec,lo_hi_rec,hi_lo_rec,hi_hi_rec],
                           axis=2)
    filter_bank_rec = tf.expand_dims(filter_bank_rec,2)
    
    return filter_bank_dec, filter_bank_rec

def fwt(f_img, 
        filter_bank, 
        scale = 1, 
        pad_mode = 'REFLECT',
        output_type = 'image',
        method = "strided"):
    """
    Implements the fast wavelet transform for orthogonal or
    biorthogonal wavelets.
    f_img -- Tensorflow Tensor. The input image.
    filter_bank -- Tensorflow Tensor. The filter_bank.
    scale -- int. The scale up to which the FWT is computed.
    pad_mode -- 'REFLECT', 'SYMMETRIC' or 'CONSTANT' for boundary
                handling.
    output_type -- 'image' or 'list'. 'list' in incompatible with
                    multi_channel_FWT so far.
    method -- 'strided' or 'downsampling'. Only for troubleshooting.
    """
    
    # The convolution and subsequent subsampling is realized
    # through a strided convolution with padding.
    filter_shape = filter_bank.get_shape().as_list()
    h = filter_shape[0]//2-1
    w = filter_shape[1]//2-1
    
    
    if method == "strided":
        f_img = tf.pad(f_img,[[0,0],[h,h],[w,w],[0,0]],pad_mode)
        filtered = tf.nn.conv2d(f_img, 
                                filter_bank, 
                                strides=[1,2,2,1], 
                                padding='VALID')
    if method == "downsampling":
        # gives EXACTLY the same result as "strided" with pad_mode="CONSTANT".
        # to be deleted.
        filtered = tf.nn.conv2d(f_img,
                               filter_bank,
                               strides=[1,1,1,1],
                               padding='SAME')
        downsampling_kernel = np.zeros((2,2,4,4),dtype=np.float32)
        for i in range(4):
            downsampling_kernel[:,:,i,i] = np.array([[1,0],[0,0]])    
        filtered = tf.nn.conv2d(filtered,
                               downsampling_kernel,
                               strides=[1,2,2,1],
                               padding='SAME')
    
    

    # filtered is a tensor of dimension [N,H,W,4].
    # Turn this into 4 tensors of dimension [N,H,W,1] in
    # order to be processable by the recursive function. 
    filtered = tf.unstack(filtered, axis=-1)
    a = tf.expand_dims(filtered[0],axis=-1)
    d_1 = tf.expand_dims(filtered[1],axis=-1)
    d_2 = tf.expand_dims(filtered[2],axis=-1)
    d_3 = tf.expand_dims(filtered[3],axis=-1)

    # FWT recursion of a.
    if scale>1:
        a = fwt(a, 
                filter_bank, 
                scale-1, 
                pad_mode, 
                output_type)  
    
    if output_type == 'image':
        filtered_upper = tf.concat([a,d_2],1)
        filtered_lower = tf.concat([d_1,d_3],1)
        filtered = tf.concat([filtered_upper, filtered_lower],2)
        return filtered
    elif output_type == 'list':
        return [a,d_1,d_2,d_3]

def multi_channel_fwt(f_img, 
        filter_bank, 
        scale = 1, 
        pad_mode = 'REFLECT',
        output_type = 'image'):
    
    """
    This function implements the multi-channel FWT.
    Currently only works with 'image'
    """
    
    img_channels = tf.unstack(f_img, axis=-1)
    transformed_channels = []
    for channel in img_channels:
        transformed_channels.append(
            fwt(tf.expand_dims(channel,axis=-1), 
                filter_bank, 
                scale, 
                pad_mode,
                output_type = output_type))
        # only works with 'image' so far
    if output_type == 'image':
        combined_img = tf.stack(transformed_channels,
                               axis = -1)
        return tf.squeeze(combined_img, [3])
    if output_type == 'list':
        return transformed_channels