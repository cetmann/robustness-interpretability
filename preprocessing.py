""""
Preprocessing pipeline for all datasets.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



def image_preprocessing(file, 
                        label, 
                        file_type = 'filename',
                        shape = [224,224],
                        random_events = True,
                        data_augmentation = True,
                        additive_noise = False,
                        subtract_by = 0.,
                        divide_by = 1.,
                        colorspace = 'BGR',
                        rescale=True,
                        min_rescale=256,
                        max_rescale=512,
                        test_rescale=256,
                        noise_level=1,
                        clip_values = None):   
    """
    Implementation of a flexible image preprocessing pipeline.
    """
    if file_type == 'filename':
        image = tf.read_file(file)
        image = tf.image.decode_jpeg(image,channels=3)
        
    elif file_type == 'tfrecord':
        features = tf.parse_single_example(
            file,
            features={
                'image/height': tf.FixedLenFeature([],tf.int64),
                'image/width': tf.FixedLenFeature([],tf.int64),
                'image/colorspace': tf.FixedLenFeature([],tf.string),
                'image/channels': tf.FixedLenFeature([],tf.int64),
                'image/class/label': tf.FixedLenFeature([],tf.int64),
                'image/class/text': tf.FixedLenFeature([],tf.string),
                'image/format': tf.FixedLenFeature([],tf.string),
                'image/filename': tf.FixedLenFeature([],tf.string),
                'image/encoded': tf.FixedLenFeature([],tf.string)
            })
        height = tf.cast(features['image/height'],tf.float32)
        width = tf.cast(features['image/width'],tf.float32)
        channels = tf.cast(features['image/channels'],tf.float32)
        img_size = tf.stack([height,width,channels],axis=0)
        
        # In the TFRecord file, the labels are encoded from 1 to 1000.
        # Here, we convert to 0,...,999
        label = features['image/class/label'] - 1
        
        # Convert the serialized byte image files to tensors.
        image = features['image/encoded']
        image = tf.image.decode_jpeg(image,channels=3)
        
    elif file_type == 'tensor':
        image = file
        
    else:
        raise ValueError('file_type must be "filename" or "tensor"')
        
    image = tf.cast(image,tf.float32)    
        
    
    [d_y,d_x] = list(shape)
    

    
    if rescale:
        if random_events:
            rescale_size = tf.random_uniform((1,),
                                             min_rescale,
                                             max_rescale,
                                             dtype=tf.float32)[0]        
        else:
            rescale_size = tf.constant(test_rescale, 
                                       dtype=tf.float32)
        
        if file_type != 'tfrecord':
            
            image_size = tf.shape(image)
            height = tf.cast(image_size[0], tf.float32)
            width = tf.cast(image_size[1], tf.float32)


        h_l_w = [tf.cast(rescale_size, tf.int32),
                 tf.cast(rescale_size/height*width+1, tf.int32)]
        w_l_h = [tf.cast(rescale_size/width*height+1, tf.int32),
                 tf.cast(rescale_size, tf.int32)]
        
        new_image_size = tf.cond(tf.less(height, width),
                                 lambda: h_l_w,
                                 lambda: w_l_h)
        new_height = new_image_size[0]
        new_width = new_image_size[1]
        
        image = tf.image.resize_images(image,new_image_size)

        if random_events:
            crop_y = tf.random_uniform((1,),0,new_height-d_y,dtype=tf.int32)[0] 
            crop_x = tf.random_uniform((1,),0,new_width-d_x,dtype=tf.int32)[0] 
            crop_location = tf.cast([crop_y,crop_x,0],tf.int32)
            crop_size = tf.cast([d_y,d_x,3],tf.int32)

        else:
            crop_y = (new_height-d_y)/2
            crop_x = (new_width-d_x)/2
            crop_location = tf.cast([crop_y,crop_x,0],tf.int32)
            crop_size = tf.cast([d_y,d_x,3],tf.int32)
        
        image = tf.slice(image,crop_location,crop_size)

            
    if data_augmentation:
        image = tf.image.random_hue(image,.06)
        image = tf.image.random_flip_left_right(image)
	# Behaviour not tested yet:
	#image = tf.image.random_brightness(image,.8,1.2)
	#image = tf.image.random_contrast(image,.2)
 

    # The original implementations of VGG-nets and
    # ResNets use BGR colorspace. If we used pre-trained
    # weights, we should switch to the same colorspace.
    if colorspace == 'BGR':
        image = image[:,:,::-1]        
    image = tf.cast(image, tf.float32)
    
    if subtract_by != 0:
        if subtract_by == 'ImageNet':
            # VGG preprocessing uses this
            subtrahend = np.array([103.939,116.779,123.68])
            subtrahend = subtrahend * tf.ones_like(image)   
        else:
            subtrahend = np.array(subtract_by, dtype='float32')
        image-= subtrahend
    
    if divide_by != 1:
        image/= divide_by
    
    if additive_noise:
        # We don't add noise of a constant std
        # to the images, because that would mean that
        # (for sufficiently large images) all images
        # that the neural network sees are equally
        # noisy. Instead, we randomly vary the noise
        # level in every batch.
        # This was NOT USED in the paper.
        noise_level = tf.random_uniform((1,),
                        minval = 0,
                        maxval = tf.abs(noise_level)+1e-6)
        image+= tf.random_normal((d_y,d_x,3),
                        stddev = noise_level)
    
    # When using additive noise, the tensor values
    # can theoretically escape the original image
    # value range. This can be clipped by setting
    # clip_values = [min_value, max_value]
    if clip_values:
        image = tf.clip_by_value(image,
                                clip_values[0],
                                clip_values[1])
    
    return image, label


def interpret_as_image(image,
                      add_by=0.,
                      multiply_by=1.,
                      colorspace = 'RGB'):
    """
    Here, image is NHWC, not HWC like above.
    """
    if multiply_by !=1.:
        image*= multiply_by
    if add_by != 0.:
        if add_by == "ImageNet":
            summand = np.array([103.939,116.779,123.68])
            summand = summand * np.ones_like(image)   
        else:
            summand = np.array(subtract_by, dtype='float32')
        image+= summand
    if colorspace == 'BGR':
        image = image[:,:,:,::-1]
    return image
