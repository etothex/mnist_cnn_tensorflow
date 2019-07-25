#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @Author  : GaoYQ
# @Time    : 2019/7/9 14:10
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import argparse
import forward_dnn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def argument_parser(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--NUM_TRAIN_EXAMPLES', default='55000', type=int)
    parser.add_argument('--NUM_VALIDATION_EXAMPLES', default='5000', type=int)
    parser.add_argument('--NUM_TEST_EXAMPLES', default='10000', type=int)
    parser.add_argument('--INPUT_NODE', default=28*28, type=int)
    parser.add_argument('--OUTPUT_NODE', default=10, type=int)
    parser.add_argument('--LAYER1_NODE', default=500, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--learning_rate_base', default=0.001, type=float)
    parser.add_argument('--learning_rate_decay', default=0.99, type=float)
    parser.add_argument('--l2_loss_rate', default=0.001, type=float)
    parser.add_argument('--moving_average_decay', default=0.99, type=float)
    parser.add_argument('--training_steps', default=30000, type=int)
    parser.add_argument('--buffer_size', default=10000, type=int)
    parser.add_argument('--count', help='dataset repeat count', default=10, type=int)
    parser.add_argument('--model_save_path', default='models_logs_dnn')
    parser.add_argument('--model_name', default='mnist_model')
    parser.add_argument('--save_freq', default=100, type=int)

    return parser


def tfrecords_parser(record):
    features = tf.parse_single_example(
        serialized=record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
    reshaped_image = tf.reshape(decoded_image, [28*28])
    retyped_image = tf.cast(reshaped_image, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return retyped_image, label


def get_batch(dataset, buffer_size, count, batch_size):
    dataset = dataset.map(tfrecords_parser)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat(count=count)
    dataset = dataset.batch(batch_size=batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()
    return image_batch, label_batch


def main(argv=None):
    arguments = argument_parser(argv).parse_args()
    DATA_PATH = {'train': 'mnist_data/mnist_train.tfrecords',
                 'validation': 'mnist_data/mnist_validation.tfrecords',
                 'test': 'mnist_data/mnist_test.tfrecords'}
    # get batch
    dataset_train = tf.data.TFRecordDataset([DATA_PATH['train']])
    image_batch, label_batch = get_batch(dataset_train, arguments.buffer_size, arguments.count, arguments.batch_size)
    # calculate loss
    logits = forward_dnn.inference(image_batch,
                              arguments.INPUT_NODE,
                              arguments.LAYER1_NODE,
                              arguments.OUTPUT_NODE,
                              arguments.l2_loss_rate)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                   labels=label_batch)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(arguments.learning_rate_base,
                                               global_step,
                                               arguments.NUM_TRAIN_EXAMPLES / arguments.batch_size,
                                               arguments.learning_rate_decay,
                                               staircase=True)
    # updates the variables, increments the global_step.
    optimize_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # moving average
    ema = tf.train.ExponentialMovingAverage(decay=arguments.moving_average_decay, num_updates=global_step)
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([optimize_op, ema_op]):
        train_op = tf.no_op(name='train')

    # creates a 'Saver', save graph and variables
    saver = tf.train.Saver()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # creat a Session
    with tf.Session(config=config) as sess:
        # initialization
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(arguments.model_save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        # 进行训练，直到完成迭代次数或者读取完全部数据
        for i in range(arguments.training_steps):
            # 数据集读取完会引发异常
            try:
                _, loss_value, step = sess.run([train_op, loss, global_step])
            except tf.errors.OutOfRangeError as err:
                print(err)
                print('step:', sess.run(global_step))
                break
            if i % arguments.save_freq == 0:
                saver.save(sess=sess,
                           save_path=os.path.join(arguments.model_save_path, arguments.model_name),
                           global_step=global_step)
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))


if __name__ == '__main__':
    tf.app.run()

