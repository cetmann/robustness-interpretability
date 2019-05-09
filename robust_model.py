from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.client import device_lib
import utils
import summary_utils

from foolbox.models import TensorFlowModel
from diff_ops import Rop


def bias_shifted_input(x, b, direction):
    direction_norm_squared = tf.reduce_sum(
        direction ** 2,
        axis=[1, 2, 3],
        keepdims=True)
    return x + tf.reshape(b,[-1,1,1,1]) * direction / tf.reshape(direction_norm_squared,[-1,1,1,1])

class robust_model:
    def __init__(self,
                 iterator,
                 session,
                 model,
                 num_classes,
                 optimizer,
                 dataset,
                 p_norm = 2.,
                 alpha = None,
                 decomp_type = 'bior2.2',
                 NUMPY_images = None,
                 NUMPY_labels = None,
                 learning_rate = .001,
                 weight_decay_p = .0001,
                 lp_wavelet_p = .0001,
                 batch_size = 32,
                 bn_momentum = .99,
                 robust_regularization = True,
                 use_wavelet_decomposition = True,
                 wavelet_weights = [0,1],
                 sensitivity_mode = 'logits',
                 graph = tf.get_default_graph()):
        
        self.iterator = iterator
        self.session = session
        self.model = model
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.dataset = dataset
        self.robust_regularization = robust_regularization
        self.wavelet_weights = wavelet_weights
        self.nested_wavelet_weights = utils.nested_weight_list(
            wavelet_weights)
        self.sensitivity_mode = sensitivity_mode
        self.graph = graph
        self.decomp_type = decomp_type

        self.decomp_depth = len(wavelet_weights)-1
        self.learning_rate = learning_rate
        self.weight_decay_p = weight_decay_p
        self.lp_wavelet_p = lp_wavelet_p
        self.batch_size = batch_size
        self.bn_momentum = bn_momentum
        self.graph = tf.get_default_graph()
        self.p_norm = p_norm
        
        
        self.alpha = alpha
        self.NUMPY_images = NUMPY_images
        self.NUMPY_labels = NUMPY_labels

        if use_wavelet_decomposition:
            from fwt import multi_channel_fwt, create_filter_bank
            self.decomp_filters, self.reconst_filters = create_filter_bank(
                decomp_type)

        devices = device_lib.list_local_devices()
        GPU_devices = [dev.name for dev in devices 
                       if dev.device_type=='GPU']
        self.num_GPUs = len(GPU_devices)
        
        tensors = []
        scalars = []
        gradients = []
        summaries = []
        with tf.variable_scope(tf.get_variable_scope()):
            with session.as_default():
                for dev in range(self.num_GPUs):
                    with tf.device('/device:GPU:%d' % dev):               
                        with tf.name_scope('GPU_%d' % dev) as scope:
                            print("Compiling on GPU %d ..." %dev)

                            tensors.append(dict())
                            scalars.append(dict())

                            # scalars finished converting to dict:
                            # mean_NLL, sum_of_true_logits, mean_correlations

                            # Get the inputs from the iterators
                            next_element = iterator.get_next()
                            tensors[-1]['images'] = next_element[0]
                            tensors[-1]['targets'] = next_element[1]
                            tensors[-1]['one_hot_targets'] = tf.one_hot(
                                tensors[-1]['targets'],
                                self.num_classes)

                            # Get the forward propagated output
                            # for the current batch of this GPU.
                            network_output = model(tensors[-1]['images'])
                            tensors[-1]['logits'] = network_output
                            
                                                     
                            

                            # For neural networks that use batch
                            # normalization, network_output is actually
                            # a list of tensors, where logits[1:]
                            # represent the inputs to the BatchNorm
                            # layers. Here, we handle this situation
                            # if it arises.
                            if type(network_output) == list:
                                tensors[-1]['logits'] = network_output[0]
                                bn_inputs = network_output[1:]
                                utils.add_bn_ops(model,
                                                 bn_inputs,
                                                 bn_momentum=bn_momentum)
                            
                            
                            tensors[-1]['predictions'] = tf.argmax(
                                tensors[-1]['logits'],
                                axis=1)
                            tensors[-1]['predicted_one_hot_targets'] = tf.one_hot(
                                tensors[-1]['predictions'],
                                self.num_classes)
                            tensors[-1]['predicted_logits'] = tf.reduce_max(
                                tensors[-1]['logits'],
                                axis=1)
                            tensors[-1]['probabilities'] = tf.nn.softmax(
                                tensors[-1]['logits'])


                            
                            #### x-terms, b-terms ####################
                            
                            tensors[-1]['x_terms'] = Rop(tensors[-1]['logits'],
                                                         tensors[-1]['images'],
                                                         tensors[-1]['images'])
                            tensors[-1]['b_terms'] = tensors[-1]['logits'] - tensors[-1]['x_terms']
                            tensors[-1]['predicted_b_terms'] = utils.select(tensors[-1]['b_terms'],
                                                                tensors[-1]['predictions'],
                                                                self.num_classes)
                            
                            if self.alpha is not None:
                                tensors[-1]['taus'] = tensors[-1]['logits'] - self.alpha * tensors[-1]['x_terms']
                                

                            #NUMPY SECTION
                            if NUMPY_images is not None and NUMPY_labels is not None:
                                NUMPY_network_output = model(NUMPY_images)
                                tensors[-1]['NUMPY_logits'] = NUMPY_network_output
                                if type(NUMPY_network_output) == list:
                                    tensors[-1]['NUMPY_logits'] = NUMPY_network_output[0]
                                tensors[-1]['NUMPY_predictions'] = tf.argmax(
                                    tensors[-1]['NUMPY_logits'],
                                    axis=1)

                                tensors[-1]['NUMPY_x_terms'] = Rop(tensors[-1]['NUMPY_logits'],
                                                                   NUMPY_images,
                                                                   NUMPY_images)
                                tensors[-1]['NUMPY_b_terms'] = tensors[-1]['NUMPY_logits'] - tensors[-1]['NUMPY_x_terms']


                                tensors[-1]['NUMPY_selected_x_terms'] = utils.select(
                                    tensors[-1]['NUMPY_x_terms'],
                                    NUMPY_labels,
                                    self.num_classes)
                                tensors[-1]['NUMPY_selected_b_terms'] = utils.select(
                                    tensors[-1]['NUMPY_b_terms'],
                                    NUMPY_labels,
                                    self.num_classes)

                                if self.alpha is not None:
                                    NUMPY_taus = tensors[-1]['NUMPY_logits'] - self.alpha * tensors[-1]['NUMPY_x_terms']

                                tensors[-1]['NUMPY_selected_logits'] = utils.select(
                                    tensors[-1]['NUMPY_logits'],
                                    NUMPY_labels,
                                    self.num_classes)

                                tensors[-1]['NUMPY_logit_sensitivities'] = tf.gradients(
                                    tf.reduce_sum(tensors[-1]['NUMPY_selected_logits']),
                                    NUMPY_images)[0]
                                tensors[-1]['NUMPY_bias_shifted_images'] = bias_shifted_input(
                                    NUMPY_images,
                                    tensors[-1]['NUMPY_selected_b_terms'],
                                    tensors[-1]['NUMPY_logit_sensitivities'])

                            
                            
                            ##########################################
                                                    

                            # Classification loss
                            tensors[-1]['NLLs'] = tf.nn.softmax_cross_entropy_with_logits_v2(
                                labels = tensors[-1]['one_hot_targets'],
                                logits = tensors[-1]['logits']
                                )
                            scalars[-1]['mean_NLL'] = tf.reduce_mean(tensors[-1]['NLLs'])

                            # Setting up the sensitivity penalty.
                            if sensitivity_mode == 'logits':                    
                                scalars[-1]['sum_of_true_logits'] = tf.reduce_sum(
                                    tensors[-1]['logits'] * tensors[-1]['one_hot_targets'])
                                tensors[-1]['sensitivities'] = tf.gradients(
                                    scalars[-1]['sum_of_true_logits'],
                                    tensors[-1]['images'],
                                    name='input_gradients')[0]
                            elif sensitivity_mode == 'NLL':
                                tensors[-1]['sensitivities'] = tf.gradients(
                                    scalars[-1]['mean_NLL'],
                                    tensors[-1]['images'],
                                    name='input_gradients')[0]

                                
                            if use_wavelet_decomposition:
                                sensitivity_w_decomp = multi_channel_fwt(
                                    tensors[-1]['sensitivities'],
                                    self.decomp_filters,
                                    self.decomp_depth,
                                    output_type = 'list')


                            tensors[-1]['inner_products'] = tf.reduce_sum(
                                tensors[-1]['images'] * tensors[-1]['sensitivities'],
                                axis = [1,2,3])

                            tensors[-1]['sensitivity_norms'] = tf.sqrt(tf.reduce_sum(
                                tensors[-1]['sensitivities']**2,
                                axis=[1,2,3],
                                name='sens_norm'))
                            tensors[-1]['image_norms'] = tf.sqrt(tf.reduce_sum(
                                tensors[-1]['images']**2,
                                axis=[1,2,3],
                                name='im_norm'))

                            tensors[-1]['norm_products'] = tensors[-1]['sensitivity_norms'] * tensors[-1]['image_norms']
                            
                            epsilon = 0.0
                            tensors[-1]['correlations'] = tensors[-1]['inner_products'] / (
                                    tensors[-1]['norm_products'] + epsilon)
                            
                            scalars[-1]['mean_correlation'] = tf.reduce_mean(tensors[-1]['correlations'])
                            scalars[-1]['mean_inner_product'] = tf.reduce_mean(tensors[-1]['inner_products'])
                            scalars[-1]['mean_norm_product'] = tf.reduce_mean(tensors[-1]['norm_products'])
                            
                            
                            tensors[-1]['true_logits'] = tf.reduce_sum(
                                tensors[-1]['logits'] * tensors[-1]['one_hot_targets'],axis=1)
                            scalars[-1]['sum_of_true_logits'] = tf.reduce_sum(
                                tensors[-1]['true_logits'])
                            tensors[-1]['logit_sensitivities'] = tf.gradients(
                                scalars[-1]['sum_of_true_logits'],
                                tensors[-1]['images'],
                                name='logit_input_gradients')[0]
                            
                            tensors[-1]['logit_inner_products'] = tf.reduce_sum(
                                tensors[-1]['images'] * tensors[-1]['logit_sensitivities'],
                                axis = [1,2,3])

                            tensors[-1]['logit_sensitivity_norms'] = tf.sqrt(tf.reduce_sum(
                                tensors[-1]['logit_sensitivities']**2,
                                axis=[1,2,3],
                                name='sens_norm'))
                            
                            tensors[-1]['logit_norm_products'] = tensors[-1]['logit_sensitivity_norms'] * tensors[-1]['image_norms']
                            
                            tensors[-1]['logit_correlations'] = tensors[-1]['logit_inner_products'] / \
                                (tensors[-1]['logit_norm_products'] + epsilon)
                            
                            scalars[-1]['mean_logit_correlation'] = tf.reduce_mean(tensors[-1]['logit_correlations'])
                            scalars[-1]['mean_logit_inner_product'] = tf.reduce_mean(tensors[-1]['logit_inner_products'])
                            scalars[-1]['mean_logit_norm_product'] = tf.reduce_mean(tensors[-1]['logit_norm_products'])
                            

                            
                            
                            # Again as a tiled image, for visualization.
                            # Only do this if the dimensions work out.
                            tiled_image_works = False
                            if use_wavelet_decomposition:
                                try:
                                    tensors[-1]['sensitivity_w_decomp_imgs'] = multi_channel_fwt(
                                        tensors[-1]['sensitivities'],
                                        self.decomp_filters,
                                        self.decomp_depth,
                                        output_type = 'image')
                                    tiled_image_works = True
                                except tf.errors.OpError:
                                    print("Creating a tiled wavelet image failed.")
                                    

                            # sum up all the p-norms of the FWTs of
                            # all channels.
                            if use_wavelet_decomposition:
                                sensitivity_w_mean_lp = 0
                                for decomp in sensitivity_w_decomp:
                                    sensitivity_w_mean_lp+= utils.lp_norm_weighted(
                                        decomp,
                                        self.nested_wavelet_weights,
                                        p_norm = self.p_norm)
                            else:
                                # Otherwise, just calculate the p-norm of the
                                # sensitivity.
                                sensitivity_w_mean_lp = utils.lp_norm(tensors[-1]['sensitivities'],
                                                            p_norm = self.p_norm)

                            scalars[-1]['sensitivity_w_mean_lp'] = sensitivity_w_mean_lp
                            
                            
                            ############ ONLY FOR LOGGING PURPOSES ###################
                            tensors[-1]['random_targets'] = tf.random_uniform(tf.shape(tensors[-1]['targets']),
                                                              maxval = self.num_classes-1,
                                                              dtype=tf.int32)

                            tensors[-1]['random_one_hot_targets'] = tf.one_hot(
                                tensors[-1]['random_targets'],
                                self.num_classes)
                            tensors[-1]['random_logits'] = tf.reduce_sum(
                                tensors[-1]['logits'] * tensors[-1]['random_one_hot_targets'],
                                axis=1)
                            scalars[-1]['sum_of_random_logits'] = tf.reduce_sum(
                                tensors[-1]['random_logits'])
                            
                            tensors[-1]['random_logit_sensitivities'] = tf.gradients(
                                scalars[-1]['sum_of_random_logits'],
                                tensors[-1]['images'],
                                name='random_logit_sensitivities')[0]
                            tensors[-1]['random_logit_inner_products'] = tf.reduce_sum(
                                tensors[-1]['images']*tensors[-1]['random_logit_sensitivities'],
                                axis=[1,2,3])
                            tensors[-1]['random_logit_sensitivity_norms'] = tf.sqrt(tf.reduce_sum(
                                tensors[-1]['random_logit_sensitivities']**2,
                                axis=[1,2,3]))
                            
                            
                            scalars[-1]['sum_of_predicted_logits'] = tf.reduce_sum(
                                tensors[-1]['predicted_logits'])
                            tensors[-1]['predicted_logit_sensitivities'] = tf.gradients(
                                scalars[-1]['sum_of_predicted_logits'],
                                tensors[-1]['images'],
                                name='predicted_logit_sensitivities')[0]
                            tensors[-1]['predicted_logit_inner_products'] = tf.reduce_sum(
                                tensors[-1]['images']*tensors[-1]['predicted_logit_sensitivities'],
                                axis=[1,2,3])
                            tensors[-1]['predicted_logit_sensitivity_norms'] = tf.sqrt(tf.reduce_sum(
                                tensors[-1]['predicted_logit_sensitivities']**2,
                                axis=[1,2,3]))

                            tensors[-1]['true_logit_sensitivities'] = tensors[-1]['logit_sensitivities']
                            tensors[-1]['true_logit_inner_products'] = tf.reduce_sum(
                                tensors[-1]['images'] * tensors[-1]['true_logit_sensitivities'],
                                axis = [1,2,3])
                            tensors[-1]['true_logit_sensitivity_norms'] = tf.sqrt(tf.reduce_sum(
                                tensors[-1]['true_logit_sensitivities']**2,
                                axis=[1,2,3]))
                            

                            
                            # Calculate the bias gradients
                            flatten = lambda a : tf.reshape(a,(-1,))
                            IP = lambda a,b : tf.reduce_sum(a*b)                  

                            biases = [b for b in model.trainable_weights if 'bias' in b.name]
                            biases+= tf.get_collection('bn_betas')
                            biases+= tf.get_collection('bn_means')

                            random_bias_gradients = tf.gradients(
                                scalars[-1]['sum_of_random_logits'],
                                biases, 
                                name='random_bias_gradients')

                            
                            random_bg = [IP(flatten(b),flatten(g)) for (b,g) in zip(biases, random_bias_gradients)]
                            random_bias_inner_products = tf.accumulate_n(random_bg)

                            predicted_bias_gradients = tf.gradients(
                                scalars[-1]['sum_of_predicted_logits'],
                                biases, 
                                name='predicted_bias_gradients')
                            predicted_bg = [IP(flatten(b),flatten(g)) for (b,g) in zip(biases, predicted_bias_gradients)]
                            predicted_bias_inner_products = tf.accumulate_n(predicted_bg)

                            true_bias_gradients = tf.gradients(
                                scalars[-1]['sum_of_true_logits'],
                                biases, 
                                name='true_bias_gradients')
                            
                            
                            true_bg = [IP(flatten(b),flatten(g)) for (b,g) in zip(biases, true_bias_gradients)]
                            true_bias_inner_products = tf.add_n(true_bg)
                            
                            zero_image = tf.zeros_like(tensors[-1]['images'])
                            tensors[-1]['zero_output'] = model(zero_image)[0]

                            tensors[-1]['random_zero_logits'] = tf.reduce_sum(
                                tensors[-1]['zero_output'] * tensors[-1]['random_one_hot_targets'],
                                axis=1)
                            tensors[-1]['predicted_zero_logits'] = tf.reduce_sum(
                                tensors[-1]['zero_output'] * tensors[-1]['predicted_one_hot_targets'],
                                axis=1)
                            tensors[-1]['true_zero_logits'] = tf.reduce_sum(
                                tensors[-1]['zero_output'] * tensors[-1]['one_hot_targets'],
                                axis=1)
                            
                            
                            
                            # Calculate the approximate random robustness

                            tensors[-1]['inner_product_differences'] = (tensors[-1]['predicted_logit_inner_products'] -
                                                                        tensors[-1]['random_logit_inner_products'])

                            tensors[-1]['bias_differences'] = predicted_bias_inner_products - random_bias_inner_products

                            numerator = tensors[-1]['inner_product_differences'] - tensors[-1]['bias_differences']
                            
                            tensors[-1]['logit_sensitivity_differences'] = (
                                    tensors[-1]['predicted_logit_sensitivities'] -
                                    tensors[-1]['random_logit_sensitivities'])
                            denominator = tf.sqrt(tf.reduce_sum(tensors[-1]['logit_sensitivity_differences']**2))

                            tensors[-1]['approximate_random_robustness'] = numerator/denominator
                            tensors[-1]['inner_product_differences_normalized'] = (
                                    tensors[-1]['inner_product_differences'] / denominator)
                            tensors[-1]['bias_differences_normalized'] = tensors[-1]['bias_differences'] / denominator

                            tensors[-1]['bias_difference_shifted_images'] = bias_shifted_input(
                                tensors[-1]['images'],
                                tensors[-1]['bias_differences'],
                                tensors[-1]['logit_sensitivity_differences'])


                            #print(tensors[-1]['bias_differences_normalized'])
                            #crash()
                            #######################################################



                            # Collect the network's weights and set up
                            # the weight decay penalty
                            trainable_weights = model.trainable_weights
                            scalars[-1]['weight_norm'] = tf.add_n(
                                [tf.reduce_sum(w**2) for w in trainable_weights])

                            # Assemble the total loss for this GPU
                            scalars[-1]['total_loss'] = scalars[-1]['mean_NLL']
                            scalars[-1]['total_loss']+= weight_decay_p * scalars[-1]['weight_norm']
                            if robust_regularization:
                                scalars[-1]['sensitivity_penalty'] = lp_wavelet_p * scalars[-1]['sensitivity_w_mean_lp']
                                scalars[-1]['total_loss']+= scalars[-1]['sensitivity_penalty']

                            # Everything that is tracked during training
                            # goes here. Top-5 and top-1 accuracies are 
                            # automatically added.
                            summary_dict={
                                'total_loss': scalars[-1]['total_loss'],
                                'mean_NLL': scalars[-1]['mean_NLL'],
                                'weight_2_norm_squared': scalars[-1]['weight_norm'],
                                'mean_sensitivity_wavelet_coeffs_lp': scalars[-1]['sensitivity_w_mean_lp']}

                            # Add some hyperparameters, too.
                            # Some redundant calculations through averaging
                            # later, but the computational overhead is negligible.
                            summary_dict['learning_rate_'] = learning_rate
                            summary_dict['correlation_'] = scalars[-1]['mean_correlation']
                            summary_dict['inner_product_'] = scalars[-1]['mean_inner_product']
                            summary_dict['norm_product_'] = scalars[-1]['mean_norm_product']
                            summary_dict['logit_correlation_'] = scalars[-1]['mean_logit_correlation']
                            summary_dict['logit_inner_product_'] = scalars[-1]['mean_logit_inner_product']
                            summary_dict['logit_norm_product_'] = scalars[-1]['mean_logit_norm_product']
                            summary_dict['weight_decay_parameter_'] = weight_decay_p
                            summary_dict['lp_Wavelet_parameter_'] = lp_wavelet_p
                            summary_dict['total_batch_size'] = batch_size * self.num_GPUs
                            summary_dict['bn_momentum_'] = bn_momentum
                            summary_dict['p_norm'] = p_norm

                            if robust_regularization:
                                summary_dict['sensitivity_penalty'] = scalars[-1]['sensitivity_penalty']
                                

                            summary_dict = summary_utils.prepare_summaries(
                                summary_dict = summary_dict,
                                predictions = tensors[-1]['probabilities'],
                                labels = tensors[-1]['targets'])
                            summaries.append(summary_dict)

                            # Collect the gradients for every GPU
                            gradients.append(
                                optimizer.compute_gradients(
                                    scalars[-1]['total_loss'],
                                    var_list=trainable_weights,
                                    colocate_gradients_with_ops=True))

                            # So far, the adversarial attack model is only
                            # created on one GPU. Different parallelized versions
                            # always lead to errors.
                            if dev == 0:
                                self.adversarial_model = TensorFlowModel(
                                    tensors[-1]['images'],
                                    tensors[-1]['logits'], 
                                    bounds=self.dataset.bounds)
                


        print("Done.")

        # Copy the lists 'tensors' and 'scalars' and replace these with an aggregated version:
        # Concatenate the tensors and average the scalars.
        self.tensors = dict()
        self.scalars = dict()
        for key in tensors[0].keys():
            print(key)
            self.tensors[key] = tf.concat(
                [tensors_item[key] for tensors_item in tensors],
                axis=0)
        for key in scalars[0].keys():
            self.scalars[key] = tf.reduce_mean(
                [scalars_item[key] for scalars_item in scalars])

        # Create self.GPU_collections for backwards compatibility
        self.GPU_collections = {**self.tensors, **self.scalars}
        self.GPU_collections['top_1'] = tf.concat(
            tf.get_collection('top_1'),0)
        self.GPU_collections['top_5'] = tf.concat(
            tf.get_collection('top_5'),0)

        
        # Collection and apply the gradients over all used
        # GPUs for synchronous parallel training.
        avg_grads = utils.average_gradients(gradients)
        gradient_application = optimizer.apply_gradients(avg_grads)
        # We combine the gradient update and possibly the 
        # batch normalization update operators into one.
        self.train_op = tf.group(gradient_application,
                           *(tf.get_collection('bn_update_ops')))
        
        summary_dict = summary_utils.collect_summaries(
            summaries)
        self.summary_op = summary_utils.create_summary_op(
            summary_dict)
        
        if use_wavelet_decomposition:
            wavelet_summary = tf.summary.tensor_summary('wavelet_weights',
                                 self.wavelet_weights)
            self.summary_op = tf.summary.merge([self.summary_op,
                                               wavelet_summary])
        
        # Here, we create a tiled image summary for Tensorboard.
        # We hereby shift the range of the sensitivity and 
        # possibly its decomposition to the range of the image.
        image_range = self.dataset.image_range()
        image_max = image_range[1]
        image_min = image_range[0]
        image_span = image_max - image_min
        image_mid = image_span / 2.


        self.images = self.dataset.interpret_as_image(
            self.GPU_collections['images'])
        self.saliencies = self.GPU_collections['sensitivities']
        saliencies_max = tf.reduce_max(tf.abs(self.saliencies),
                                       [1,2],
                                       keepdims=True)
        normalized_saliencies = image_span * self.saliencies / \
            (2*saliencies_max + 1e-9) + image_mid

        if use_wavelet_decomposition:
            self.saliency_decomps = self.GPU_collections[
                'sensitivity_w_decomp_imgs']
            saliency_decomps_max = tf.reduce_max(
                tf.abs(self.saliency_decomps),
                [1,2],
                keepdims=True)
            normalized_decomps = image_span * self.saliency_decomps / \
                (2*saliency_decomps_max + 1e-9) + image_mid


        composite_image = [self.images,
             normalized_saliencies]
        
        if tiled_image_works:
            composite_image.append(normalized_decomps)
        
        
        img_saliency_decomp = tf.concat(
            composite_image,
            2)

        self.img_summary_op = tf.summary.image(
            'img_saliency_decomp',
            img_saliency_decomp,
            max_outputs = 10)
        

