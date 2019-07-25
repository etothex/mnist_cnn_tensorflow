#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @Author  : GaoYQ
# @Time    : 2019/7/9 14:09
# @Software: PyCharm

import tensorflow as tf


def get_weight(shape, l2_loss_rate=None):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if l2_loss_rate is not None:
        tf.add_to_collection('losses', l2_loss_rate * tf.nn.l2_loss(weights))
    return weights


def inference(inputs, input_node, layer1_node, output_node, l2_loss_rate):
    with tf.variable_scope('layer1'):
        weights = get_weight([input_node, layer1_node], l2_loss_rate)
        biases = tf.get_variable('biases', [layer1_node], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(inputs, weights) + biases)
    with tf.variable_scope('layer2'):
        weights = get_weight([layer1_node, output_node], l2_loss_rate)
        biases = tf.get_variable("biases", [output_node], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)
    return layer2


if __name__ == '__main__':
    pass
