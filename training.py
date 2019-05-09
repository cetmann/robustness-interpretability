"""
This class defines the training and validation pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import os

import foolbox as fb
from foolbox.criteria import Misclassification
from foolbox.adversarial import Adversarial
from foolbox.distances import Linfinity, MSE

class training:   
    def __init__(self,
                 handle,
                 dataset,
                 train_op,
                 session,
                 epoch_step,
                 batch_step,
                 summary_writer,
                 train_summary_op,
                 img_summary_op,
                 optimizer,
                 GPU_collections,
                 batch_size_placeholder,
                 pretrained = False,
                 adversarial_model = None,
                 adversarial_attacks = None,
                 adversarial_criterion = Misclassification(),
                 saver_path = "model.ckpt",
                 num_adversarial_batches = 4,
                 batch_size = 32,
                 num_epochs = 1000,
                 train_summary_period = 1000,
                 val_summary_period = 1000,
                 adv_summary_period = 1000):

        self.session = session
        
        self.saver_path = saver_path
        
        self.epoch = 0
        self.batch_i = 0
        self.handle = handle
        self.dataset = dataset
        self.train_op = train_op
        self.epoch_step = epoch_step
        self.epoch_step_increment = self.epoch_step.assign_add(1)
        self.batch_step = batch_step
        self.batch_placeholder = tf.placeholder(tf.int32,(),
                                               'b_ph')
        self.batch_step_assign = tf.assign(self.batch_step,
                                          self.batch_placeholder)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self.optimizer = optimizer
        self.GPU_collections = GPU_collections
        self.batch_size_placeholder = batch_size_placeholder
        
        # summary ops
        self.train_summary_op = train_summary_op
        self.img_summary_op = img_summary_op
        self.train_summary_period = train_summary_period
        self.val_summary_period = val_summary_period
        self.adv_summary_period = adv_summary_period
        self.summary_writer = summary_writer
        
        # validation           
        self.val_top_one_mean = tf.placeholder(
                tf.float32, name='val_top_one_mean')
        self.val_top_five_mean = tf.placeholder(
                tf.float32, name='val_top_five_mean')
        val_summaries = []
        val_summaries.append(tf.summary.scalar(
                'top_1_accuracy_validation', 
                self.val_top_one_mean))
        val_summaries.append(tf.summary.scalar(
                'top_5_accuracy_validation', 
                self.val_top_five_mean))
        self.val_summary_op = tf.summary.merge(
                val_summaries,
                name = 'val_summaries_op')
            
        # Adversarial attacks
        self.num_adversarial_batches = num_adversarial_batches
        self.adversarial_criterion = adversarial_criterion
        
        self.adv_result = tf.placeholder(
                tf.float32, name='adv_results')
        self.adversarial_attacks = adversarial_attacks
        self.adversarial_model = adversarial_model
        
        default_distances = {
            'GradientAttack' : MSE,
            'FGSM' : MSE,
            'LinfinityBasicIterativeAttack' : Linfinity,
            'L2BasicIterativeAttack' : MSE,
            'LinfinityBasicIterativeAttack' : Linfinity,
            'ProjectedGradientDescentAttack' : Linfinity,
            'DeepFoolAttack' : MSE,
            'DeepFoolLinfinityAttack' : Linfinity}

        self.attacks = dict()
        self.distances = dict() # add support for custom distances
        self.adv_summaries = dict()


        for attack in self.adversarial_attacks:
            self.attacks[attack] = getattr(fb.attacks,
                                     attack)()
            if attack in default_distances.keys():
                self.distances[attack] = default_distances[attack]
            else:
                self.distances[attack] = MSE
            
            key = attack + '_median_dist'
            
            self.adv_summaries[attack] = tf.summary.scalar(
                   attack + '_median_dist',
                   self.adv_result)  
               
        devices = device_lib.list_local_devices()
        GPU_devices = [dev.name for dev in devices 
                       if dev.device_type=='GPU']
        self.num_GPUs = len(GPU_devices)

        self.pretrained = pretrained
        

        if self.dataset.train_handle is None:
            self.dataset.get_train_handle(self.session)
            
        self.saver = tf.train.Saver(tf.global_variables())   
            
    def train(self,
             training_feed_dict,
             val_feed_dict = {},
             do_not_reload_checkpoint = False,
             do_not_save = False):  
        if (os.path.isfile(self.saver_path + '.index') and not
            do_not_reload_checkpoint):
            self.restore_model()
            self.epoch = self.session.run(self.epoch_step)
            self.batch_i = self.session.run(self.batch_step)+1
        elif not self.pretrained:
            print("Initializing variables...")
            self.session.run(tf.global_variables_initializer())
            print("Done.")
        else:
            # When starting from a pretrained network,
            # only initialize the variables that we potentially added 
            # when we implemented the parallelized batch normalization
            # update operators.
            print("Initializing batch normalization variables...")
            for var in tf.global_variables():
               if "biased" in var.name or "local_step" in var.name:
                   tf.add_to_collection('uninitialized_variables',var)
            tf.add_to_collection('uninitialized_variables',self.epoch_step)
            tf.add_to_collection('uninitialized_variables',self.batch_step)
            uninitialized_vars = tf.get_collection('uninitialized_variables')
            self.session.run(tf.variables_initializer(
                uninitialized_vars))
            print("Done.")
            self.session.run(tf.variables_initializer(
                self.optimizer.variables()))
            
            if val_feed_dict:
                self.validate(val_feed_dict)
                self.adversarial(self.adversarial_attacks,
                    val_feed_dict,
                    num_batches = self.num_adversarial_batches)    
                
        if self.dataset.train_handle is None:
            self.dataset.get_train_handle(self.session)
        training_feed_dict[self.handle] = self.dataset.train_handle
        
        
        path = self.saver_path
        base, ext = os.path.splitext(path)
        new_path = base + '_' + str(self.batch_i) + ext
        self.saver.save(self.session, 
                        new_path)
        
        print("Beginning training...")
        if self.epoch >= self.num_epochs:
            print("End of training reached.")
        while self.epoch < self.num_epochs:
            try:
                self.dataset.initialize_train_batch_iterator(
                    self.session,
                    self.batch_size)
                while True:
                    try:
                        train_output = self.session.run(
                            self.train_op,
                            training_feed_dict)

                        self.batch_i+= 1
                        if self.batch_i % self.train_summary_period == 0:
                            train_output, summary_str = self.session.run(
                                [self.train_op,self.train_summary_op],
                                training_feed_dict)
                            self.summary_writer.add_summary(summary_str, 
                                                       self.batch_i)
                            self.summary_writer.flush()
                            self.update_batch_step()
                            if not do_not_save:
                                self.save_model()
                        if (self.batch_i % self.val_summary_period == 0
                             and val_feed_dict): 
                            self.validate(val_feed_dict,
                                         do_not_save = do_not_save)
                            
                        if (self.batch_i % self.adv_summary_period == 0
                             and val_feed_dict): 
                            self.adversarial(self.adversarial_attacks,
                                val_feed_dict,
                                num_batches = self.num_adversarial_batches,
                                do_not_save = do_not_save)
                    except tf.errors.OutOfRangeError:
                        self.epoch+= 1
                        self.session.run(
                            self.epoch_step_increment)
                        break
            except KeyboardInterrupt:
                print('\nCancelled')
                break
        
        print("Training finished.")
        if val_feed_dict:
            print("Performing final validation...")
            self.validate(val_feed_dict)
            print("Performing final adversarial tests...")
            self.adversarial(self.adversarial_attacks,
                val_feed_dict,
                num_batches = self.num_adversarial_batches)
            print("Done.")
        
        if not do_not_save:
            print("Saving completed model...")
            self.saver.save(self.session, self.saver_path)
            print("Successfully saved completed model.")

            
        
        
    def validate(self, val_feed_dict, do_not_save = False):
        
        self.dataset.initialize_val_batch_iterator(self.session,
                                                   self.batch_size)

        val_feed_dict[self.handle] = self.dataset.val_handle
        
        
        # Calculate validation error #
        val_in_top_five = np.zeros(self.dataset.num_val_samples,
            np.float32)
        val_in_top_one = np.zeros(self.dataset.num_val_samples,
            np.float32)
        step_size = self.batch_size * self.num_GPUs
        
        l_index = u_index = 0
        while u_index < self.dataset.num_val_samples:
            try:
                top_fives, top_ones = self.session.run(
                    [self.GPU_collections['top_5'],
                     self.GPU_collections['top_1']],
                            feed_dict=val_feed_dict)   
                n_images = len(top_fives)
                u_index = l_index + n_images
                val_in_top_five[l_index:u_index] = top_fives 
                val_in_top_one[l_index:u_index] = top_ones
                l_index = u_index 
            except tf.errors.OutOfRangeError:
                # This handles an error that only appears when
                # there are 2 GPUs with batch size 16 each...
                # More documentation to follow.
                break
            
        val_in_top_five = val_in_top_five[:u_index]
        val_in_top_one = val_in_top_one[:u_index]   

        val_top_five_accuracy = sum(
            val_in_top_five)/np.float32(
                self.dataset.num_val_samples)
        val_top_one_accuracy = sum(
            val_in_top_one)/np.float32(
                self.dataset.num_val_samples)

        val_summary = self.session.run(self.val_summary_op, 
            feed_dict={self.val_top_one_mean : val_top_one_accuracy,
                       self.val_top_five_mean : val_top_five_accuracy})
        self.summary_writer.add_summary(val_summary,
                                        self.batch_i)

        
        
        # IMG SUMMARIES
        self.dataset.initialize_img_batch_iterator(self.session,
                            self.batch_size)      
        val_feed_dict[self.handle] = self.dataset.img_handle
        img_summary_str = self.session.run(self.img_summary_op,
            feed_dict=val_feed_dict)
        self.summary_writer.add_summary(img_summary_str,
                                        self.batch_i)
        self.summary_writer.flush()
        self.update_batch_step()   
        if not do_not_save:
            self.save_model()
        
        
    def adversarial(self,
                   adversarial_attacks,
                   adv_feed_dict,
                   distances_dict = {},
                   num_batches = 4,
                   do_not_save = False):  
        results = dict()
        for attack in adversarial_attacks:
            results[attack] = []
                
        self.dataset.initialize_img_batch_iterator(self.session,
                            self.batch_size)      
        adv_feed_dict[self.handle] = self.dataset.img_handle
        
        for run in range(num_batches):
            [images, labels] = self.session.run(
                [self.GPU_collections['images'],
                 self.GPU_collections['predictions']],
                    feed_dict=adv_feed_dict) 

            for attack_name in adversarial_attacks:
                attack = self.attacks[attack_name]
                for i in range(len(images)):
                    adversarial = Adversarial(
                        self.adversarial_model,
                        self.adversarial_criterion,
                        images[i],
                        labels[i],
                        distance = self.distances[attack_name])
                    att = attack(adversarial)
                    dist = adversarial.distance.value
                    if dist > 0:
                        results[attack_name].append(dist)
        
        for attack_name in self.adversarial_attacks:
            median_dist = np.median(results[attack_name])
            adv_summary_str = self.session.run(self.adv_summaries[attack_name],
                feed_dict = {self.adv_result : median_dist})
            
            self.summary_writer.add_summary(adv_summary_str,
                                        self.batch_i)
            
        self.summary_writer.flush() 
        self.update_batch_step()   
        if not do_not_save:
            self.save_model()
        
    def update_batch_step(self):
        self.session.run(
            self.batch_step_assign,
            feed_dict = {self.batch_placeholder : self.batch_i})

    def restore_model(self):       
        print("Trying to load old model checkpoint...")
        self.saver.restore(self.session, 
                           self.saver_path)
        print("Successfully loaded old model checkpoint.")
        
    def save_model(self):
        self.saver.save(self.session, 
                        self.saver_path)
        print("Model saved.")
