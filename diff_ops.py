"""
This file defines the L-Op and R-Op, based on https://j-towns.github.io/2017/06/12/A-new-trick.html
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def Lop(nodes, x, v):
    lop_out = tf.gradients(nodes, x, grad_ys=v)
    return lop_out


def Rop(nodes, x, v):
    if isinstance(nodes, list):
        u = [tf.ones_like(node) for node in nodes]
    else:
        u = tf.ones_like(nodes)
    rop_out = tf.gradients(
        Lop(nodes, x, u),u,grad_ys=v)
    return rop_out
