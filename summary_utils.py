""""
Some utility functions for creating the summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def prepare_summaries(summary_dict=dict(),
                      predictions=None,
                      labels=None):
    if (predictions is not None) and (labels is not None):
        top_1 = tf.nn.in_top_k(predictions,labels, 1)
        top_1 = tf.cast(top_1, tf.float32)
        tf.add_to_collection('top_1',top_1)
        top_1_acc = tf.reduce_mean(top_1)
        summary_dict['top_1_accuracy'] = top_1_acc
        
        top_5 = tf.nn.in_top_k(predictions,labels, 5)
        top_5 = tf.cast(top_5, tf.float32)
        tf.add_to_collection('top_5',top_5)
        top_5_acc = tf.reduce_mean(top_5)
        summary_dict['top_5_accuracy'] = top_5_acc
    return summary_dict

def collect_summaries(summary_dict_list):
    summary_dict = dict()
    n_dicts = np.float32(len(summary_dict_list))
    for d in summary_dict_list:
        for key in d.keys():
            # If the desired key is in the report_dict, append
            # the corresponding item to a list. Otherwise create 
            # this list.
            if key in summary_dict.keys():
                summary_dict[key].append(d[key])
            else:
                summary_dict[key] = [d[key]]
    
    for key in summary_dict.keys():
        summary_dict[key] = tf.add_n(summary_dict[key])/n_dicts
    return summary_dict

def create_summary_op(summary_dict):
    summaries = []
    for key in summary_dict.keys():
        summary = tf.summary.scalar(key, summary_dict[key])
        summaries.append(summary)
    return tf.summary.merge(summaries)