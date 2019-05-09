"""
This base class defines a dataset API which is used for both ImageNet and MNIST.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class dataset:
    
    # Method for train data shuffling
    
    def __init__(self,
                 files_ph,
                 labels_ph,
                 batch_size_ph,
                 dataset_name,
                 img_indices = None,
                 num_parallel_calls = np.int32(512)):
        # tf.placeholders go here
        self.files_ph = files_ph
        self.labels_ph = labels_ph
        self.batch_size_ph = batch_size_ph
        
        # This is a string
        self.dataset_name = dataset_name
        
        self.img_indices = img_indices
        
        self.num_parallel_calls = num_parallel_calls
        
        
        if dataset_name == 'ImageNet':
            import imagenet_data as data
        if dataset_name == 'ImageNetSingleFiles':
            import imagenet_data_from_single_files as data
        if dataset_name == 'TinyImageNet':
            import tiny_imagenet_data as data
        if dataset_name == 'MNIST':
            import mnist_data as data
            
        
        # These are either file names or tensors.
        # (Tiny)ImageNet has file names, MNIST
        # has tensors.
        self.train_data, self.train_labels = \
            data.collect_train_data()
        self.val_data, self.val_labels = \
            data.collect_val_data()
        
        self.bounds = data.bounds()
        self.image_range = data.image_range
        
        # If necessary, turn lists into arrays to allow
        # for more elaborate subindexing
        if type(self.train_data) == list : 
            self.train_data = np.array(self.train_data)        
        if type(self.train_labels) == list : 
            self.train_labels = np.array(self.train_labels)        
        if type(self.val_data) == list : 
            self.val_data = np.array(self.val_data)        
        if type(self.val_labels) == list : 
            self.val_labels = np.array(self.val_labels)
        
        self.num_train_samples = data.num_train_samples()
        self.num_val_samples = data.num_val_samples()
        
        if img_indices is None:
            self.img_indices = img_indices
            self.img_data = self.val_data
            self.img_labels = self.val_labels
        else:
            self.img_data = self.val_data[self.img_indices]
            self.img_labels = self.val_labels[self.img_indices]
        
        

        
        
        # batch iterators 
        self.train_batch_iterator = data.train_BI(
            self.files_ph,
            self.labels_ph,
            self.batch_size_ph,
            num_parallel_calls)    
        self.val_batch_iterator = data.val_BI(
            self.files_ph,
            self.labels_ph,
            self.batch_size_ph,
            num_parallel_calls)    
        self.img_batch_iterator = data.img_BI(
            self.files_ph,
            self.labels_ph,
            self.batch_size_ph,
            num_parallel_calls)   
        
        self.train_handle = None
        self.val_handle = None
        self.img_handle = None

        self.interpret_as_image = data.interpret_as_image
        
        self.n_classes = num_classes[dataset_name]

    # Shuffles (training) data on numpy-level 
    def shuffle_input(self,
                      data, 
                      labels):
        num_samples = len(labels)
        shuffle_indices = np.arange(num_samples)
        np.random.shuffle(shuffle_indices)
        labels = labels[shuffle_indices]
        data = data[shuffle_indices]   
        return data, labels
    
    def initialize_train_batch_iterator(self,
                                        session,
                                        batch_size = 16,
                                        shuffle = True):        
        self.train_data, self.train_labels = self.shuffle_input(
            self.train_data, self.train_labels)
        session.run(self.train_batch_iterator.initializer, 
             feed_dict={self.files_ph: self.train_data,
                        self.labels_ph: self.train_labels,
                        self.batch_size_ph: batch_size})
        if not self.train_handle:
            self.get_train_handle(session)

        
    def initialize_val_batch_iterator(self,
                                      session,
                                      batch_size = 16):       
        session.run(self.val_batch_iterator.initializer, 
             feed_dict={self.files_ph: self.val_data,
                        self.labels_ph: self.val_labels,
                        self.batch_size_ph: batch_size})
        if not self.val_handle:
            self.get_val_handle(session)  
            
    def initialize_img_batch_iterator(self,
                                      session,
                                      batch_size = 16):        
        session.run(self.img_batch_iterator.initializer, 
             feed_dict={self.files_ph: self.img_data,
                        self.labels_ph: self.img_labels,
                        self.batch_size_ph: batch_size})
        if not self.img_handle:
            self.get_img_handle(session)  
            
    def get_train_handle(self,session): 
        self.train_handle = session.run(
            self.train_batch_iterator.string_handle())
     
    def get_val_handle(self,session): 
        self.val_handle = session.run(
            self.val_batch_iterator.string_handle())    
        
    def get_img_handle(self,session): 
        self.img_handle = session.run(
            self.img_batch_iterator.string_handle())

    


image_resolution = {
    'ImageNet' : [256,256],
    'TinyImageNet' : [64,64],
    'MNIST' : [28,28],
    'CIFAR-10' : [32,32],
    'CIFAR-100' : [32,32]
}

num_classes = {
    'ImageNet' : 1000,
    'TinyImageNet' : 200,
    'MNIST' : 10,
    'CIFAR-10' : 10,
    'CIFAR-100' : 100
}