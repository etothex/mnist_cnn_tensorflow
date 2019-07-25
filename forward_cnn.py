#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @Author  : GaoYQ
# @Time    : 2019/7/9 20:14
# @Software: PyCharm

import tensorflow as tf

conv1_size = 5
conv1_deep = 32

conv2_size = 5
conv2_deep = 64

fc_size = 512


def inference(inputs, num_channels, num_classes, train, l2_loss_rate=None):
    # convolution layer 1, [28, 28, 1] -> [28, 28, 32]
    with tf.variable_scope('conv1'):
        conv1_weight = tf.get_variable('weight',
                                       [conv1_size, conv1_size, num_channels, conv1_deep],  # [5, 5, 1, 32]
                                       initializer=tf.truncated_normal_initializer(stddev=0.1)
                                       )
        conv1_bias = tf.get_variable('bias',
                                     [conv1_deep],  # [32]
                                     initializer=tf.constant_initializer(0.0)
                                     )
        conv1 = tf.nn.conv2d(inputs, conv1_weight, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(value=conv1, bias=conv1_bias))
    # pooling layer 1, [28, 28, 32] -> [14, 14, 32]
    with tf.variable_scope('pooling1'):
        pool1 = tf.nn.max_pool(relu1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    # convolution layer 2, [14, 14, 32] -> [14, 14, 64]
    with tf.variable_scope('conv2'):
        conv2_weight = tf.get_variable('weight',
                                       [conv2_size, conv2_size, conv1_deep, conv2_deep],    # [5, 5, 32, 64]
                                       initializer=tf.truncated_normal_initializer(stddev=0.1)
                                       )
        conv2_bias = tf.get_variable('bias',
                                     [conv2_deep],  # [64]
                                     initializer=tf.constant_initializer(0.0)
                                     )
        conv2 = tf.nn.conv2d(pool1, conv2_weight, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(value=conv2, bias=conv2_bias))
    # pooling layer 2, [14, 14, 64] -> [7, 7, 64]
    with tf.variable_scope('pooling2'):
        pool2 = tf.nn.max_pool(relu2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    # reshape image to vector
    shape_pool2 = pool2.get_shape().as_list()
    nodes = shape_pool2[1] * shape_pool2[2] * shape_pool2[3]
    pool2_reshaped = tf.reshape(pool2, [-1, nodes])
    # full connection layer 1, 7*7*64 -> 512
    with tf.variable_scope('fc1'):
        fc1_weight = tf.get_variable('weight',
                                     [nodes, fc_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只对全连接层的权重加入正则化
        if l2_loss_rate is not None:
            tf.add_to_collection('losses', l2_loss_rate * tf.nn.l2_loss(fc1_weight))
        fc1_bias = tf.get_variable('bias',
                                   [fc_size],
                                   initializer=tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(
            tf.nn.bias_add(tf.matmul(pool2_reshaped, fc1_weight), fc1_bias)
                         )
        if train is True:
            fc1 = tf.nn.dropout(fc1, 0.5)
    # full connection layer 2, 512 -> num_classes
    with tf.variable_scope('fc2'):
        fc2_weight = tf.get_variable('weight',
                                     [fc_size, num_classes],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1)
                                     )
        # 只对全连接层的权重加入正则化
        if l2_loss_rate is not None:
            tf.add_to_collection('losses', l2_loss_rate * tf.nn.l2_loss(fc2_weight))
        fc2_bias = tf.get_variable('bias',
                                   [num_classes],
                                   initializer=tf.constant_initializer(0.0)
                                   )
        logit = tf.nn.bias_add(tf.matmul(fc1, fc2_weight), fc2_bias)
    # return output
    return logit


if __name__ == '__main__':
    pass
