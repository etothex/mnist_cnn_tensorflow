#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @Author  : GaoYQ
# @Time    : 2019/7/15 0:38
# @Software: PyCharm

from multiprocessing import Process, Pool
import tensorflow as tf
import argparse
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
num_gpu = 1

num_train_examples = 55000
num_validation_examples = 5000
num_test_examples = 10000
Data_path = {'train': 'mnist_data/mnist_train.tfrecords',
             'validation': 'mnist_data/mnist_validation.tfrecords',
             'test': 'mnist_data/mnist_test.tfrecords'}


def argument_parser(argv=None):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_size', default=28, type=int)
    parser.add_argument('--image_channel', default=1, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--learning_rate_base', default=0.001, type=float)
    parser.add_argument('--learning_rate_decay', default=0.99, type=float)
    parser.add_argument('--l2_loss_rate', default=0.001, type=float)
    parser.add_argument('--moving_average_decay', default=0.99, type=float)
    parser.add_argument('--training_steps', default=30000, type=int)
    parser.add_argument('--buffer_size', default=10000, type=int)
    parser.add_argument('--count', help='dataset repeat count', default=100, type=int)
    parser.add_argument('--model_save_path', default='models_logs_cnn')
    parser.add_argument('--model_name', default='mnist_model')
    parser.add_argument('--save_interval', default=100, type=int)
    parser.add_argument('--validate_interval', default=1000, type=int)
    return parser


class ReadDataset(object):
    def __init__(self, ftrain, fval, ftest, buffer_size, count):
        self.dataset_train = tf.data.TFRecordDataset([ftrain])
        self.dataset_val = tf.data.TFRecordDataset([fval])
        self.dataset_test = tf.data.TFRecordDataset([ftest])
        self.buffer_size = buffer_size
        self.count = count

    def __del__(self):
        del self.dataset_train
        del self.dataset_val
        del self.dataset_test
        del self.buffer_size
        del self.count

    def train_batch(self, batch_size):
        dataset = self.dataset_train.map(self.tfrecords_parser)
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.repeat(count=self.count)
        dataset = dataset.batch(batch_size=batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()
        return image_batch, label_batch

    def eval_batch(self, dataset, batch_size):
        dataset = dataset.map(self.tfrecords_parser)
        dataset = dataset.batch(batch_size=batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()
        return image_batch, label_batch

    @staticmethod
    def tfrecords_parser(record):
        features = tf.parse_single_example(
            serialized=record,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'pixels': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64)
            })
        decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
        reshaped_image = tf.reshape(decoded_image, [28, 28, 1])
        retyped_image = tf.cast(reshaped_image, tf.float32)
        label = tf.cast(features['label'], tf.int32)
        return retyped_image, label


class CNNModel(object):
    def __init__(self,
                 ImageSize=28,
                 ImageChannel=1,
                 conv1_size=5,
                 conv1_deep=32,
                 conv2_size=5,
                 conv2_deep=64,
                 fc_size=512,
                 NumClasses=10):
        self.ImageSize = ImageSize
        self.ImageChannel = ImageChannel
        self.conv1_size = conv1_size
        self.conv1_deep = conv1_deep
        self.conv2_size = conv2_size
        self.conv2_deep = conv2_deep
        self.fc_size = fc_size
        self.NumClasses = NumClasses
        # build graph
        self.graph = tf.Graph()
        with self.graph.as_default() as _:
            self.x = tf.placeholder(dtype=tf.float32,
                                    shape=[None, self.ImageSize, self.ImageSize, self.ImageChannel],
                                    name='x-input')
            self.y = tf.placeholder(dtype=tf.int32, shape=[None, ], name='y-input')
            self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
            self.logit = self.inference()
            # creat session
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.session = tf.Session(config=config)
            self.summary_writer = None
            self.dataset = ReadDataset(ftrain=Data_path['train'],
                                       fval=Data_path['validation'],
                                       ftest=Data_path['test'],
                                       buffer_size=10000,
                                       count=100)
        self.logfile = ''

    def __del__(self):
        # close session
        self.session.close()
        del self.session
        del self.logit
        del self.x
        del self.y
        del self.is_training
        if self.summary_writer is not None:
            self.summary_writer.close()
        del self.summary_writer
        del self.graph

    def inference(self):
        # convolution layer 1, [28, 28, 1] -> [28, 28, 32]
        with tf.variable_scope('conv1'):
            conv1_weight = tf.get_variable('weight',
                                           [self.conv1_size, self.conv1_size, self.ImageChannel, self.conv1_deep],  # [5, 5, 1, 32]
                                           initializer=tf.truncated_normal_initializer(stddev=0.1)
                                           )
            conv1_bias = tf.get_variable('bias',
                                         [self.conv1_deep],  # [32]
                                         initializer=tf.constant_initializer(0.0)
                                         )
            conv1 = tf.nn.conv2d(self.x, conv1_weight, [1, 1, 1, 1], padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(value=conv1, bias=conv1_bias))
        # pooling layer 1, [28, 28, 32] -> [14, 14, 32]
        with tf.variable_scope('pooling1'):
            pool1 = tf.nn.max_pool(relu1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        # convolution layer 2, [14, 14, 32] -> [14, 14, 64]
        with tf.variable_scope('conv2'):
            conv2_weight = tf.get_variable('weight',
                                           [self.conv2_size, self.conv2_size, self.conv1_deep, self.conv2_deep],  # [5, 5, 32, 64]
                                           initializer=tf.truncated_normal_initializer(stddev=0.1)
                                           )
            conv2_bias = tf.get_variable('bias',
                                         [self.conv2_deep],  # [64]
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
                                         [nodes, self.fc_size],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1))
            # 只对全连接层的权重加入正则化
            tf.add_to_collection('l2_losses', tf.nn.l2_loss(fc1_weight))
            fc1_bias = tf.get_variable('bias',
                                       [self.fc_size],
                                       initializer=tf.constant_initializer(0.0))
            fc1 = tf.nn.relu(
                tf.nn.bias_add(tf.matmul(pool2_reshaped, fc1_weight), fc1_bias)
                             )
            fc1 = tf.cond(tf.equal(self.is_training, True), lambda: tf.nn.dropout(fc1, 0.5), lambda: fc1)
        # full connection layer 2, 512 -> num_classes
        with tf.variable_scope('fc2'):
            fc2_weight = tf.get_variable('weight',
                                         [self.fc_size, self.NumClasses],
                                         initializer=tf.truncated_normal_initializer(stddev=0.1)
                                         )
            # 只对全连接层的权重加入正则化
            tf.add_to_collection('l2_losses', tf.nn.l2_loss(fc2_weight))
            fc2_bias = tf.get_variable('bias',
                                       [self.NumClasses],
                                       initializer=tf.constant_initializer(0.0)
                                       )
            logit = tf.nn.bias_add(tf.matmul(fc1, fc2_weight), fc2_bias)
        # return output
        return logit

    def train(self,
              batch_size,
              learning_rate_base, learning_rate_decay,
              l2_loss_rate,
              training_steps,
              model_save_path, model_name):
        self.logfile = model_name + '-' + str(os.getpid())
        self.log(batch_size,
                 learning_rate_base, learning_rate_decay,
                 l2_loss_rate, training_steps,
                 model_save_path, model_name)
        with self.graph.as_default():
            dataset = self.dataset
            # get batch
            image_batch, label_batch = dataset.train_batch(batch_size)
            # get loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit, labels=self.y)
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            loss = cross_entropy_mean + l2_loss_rate * tf.add_n(tf.get_collection('l2_losses'))
            tf.summary.scalar('loss', loss)
            # optimization
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            learning_rate = tf.train.exponential_decay(learning_rate_base,
                                                       global_step,
                                                       num_train_examples / batch_size,
                                                       learning_rate_decay,
                                                       staircase=True)
            tf.summary.scalar('learning-rate', learning_rate)
            optimize_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # grads = optimizer.compute_gradients(loss)
            # # for grad, var in grads:
            # #     if grad is not None:
            # #         tf.summary.histogram('gradients/%s' % var.op.name, grad)
            # optimize_op = optimizer.apply_gradients(grads, global_step=global_step)
            # # for var in tf.trainable_variables():
            # #     tf.summary.histogram(var.op.name, var)

            with tf.control_dependencies([optimize_op]):
                train_op = tf.no_op(name='train')
            merge_op = tf.summary.merge_all()
            sess = self.session
            saver = tf.train.Saver()
            self.summary_writer = tf.summary.FileWriter(logdir=model_save_path, graph=sess.graph)
            # initialization
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
            # 进行训练，直到完成迭代次数或者读取完全部数据
            for i in range(training_steps):
                # 数据集读取完会引发异常
                try:
                    images, labels = sess.run([image_batch, label_batch])
                    _, loss_value, step, summary = sess.run([train_op, loss, global_step, merge_op],
                                                            feed_dict={self.x: images, self.y: labels, self.is_training: True})
                except tf.errors.OutOfRangeError as err:
                    # print(err)
                    print('dataset out of range.')
                    print('step:', sess.run(global_step))
                    break
                # save ckpt
                if (i+1) % 100 == 0:
                    saver.save(sess=sess,
                               save_path=os.path.join(model_save_path, model_name),
                               global_step=global_step)
                    self.log("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                    # print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                    self.summary_writer.add_summary(summary=summary, global_step=step)
                # validation
                if (i+1) % 1000 == 0:
                    accuracy = self.validation()
                    self.log('Validation accuracy: %g' % accuracy)
                    print("After %d training step(s), Validation accuracy is %g. Pid: %d"
                          % (step, accuracy, os.getpid()))
    
    def validation(self):
        dataset = self.dataset
        image_batch, label_batch = dataset.eval_batch(dataset.dataset_val, 100)
        accuracy = self.evaluate(image_batch, label_batch)
        return accuracy

    def test(self):
        with self.graph.as_default():
            dataset = self.dataset
            image_batch, label_batch = dataset.eval_batch(dataset.dataset_test, 100)
            accuracy = self.evaluate(image_batch, label_batch)
            self.log('Test accuracy: %g' % accuracy)
            print('Test accuracy:', accuracy, 'PID', os.getpid())

    def evaluate(self, image_batch, label_batch):
        pred_results = []
        real_labels = []
        predict_op = tf.argmax(self.logit, axis=-1, output_type=tf.int32)
        while True:
            try:
                images, labels = self.session.run([image_batch, label_batch])
                predictions = self.session.run(predict_op,
                                               feed_dict={self.x: images, self.y: labels, self.is_training: False})
                pred_results.extend(predictions)
                real_labels.extend(labels)
            except tf.errors.OutOfRangeError:
                break
        correct = [float(a == b) for (a, b) in zip(real_labels, pred_results)]
        accuracy = sum(correct) / len(correct)
        return accuracy

    def log(self, *entries):
        log_func(self.logfile, *entries)


def log_func(filename, *entries):
    with open(filename, 'a', encoding='utf-8') as f:
        if len(entries) > 1:
            f.write(str(entries))
        else:
            f.write(entries[0])
        f.write('\n')


def train_test(param):
    model = CNNModel()
    model.train(*param)
    model.test()


def main(argv=None):
    pool = Pool(4)
    hyper_para_list = [[100, 0.001, 0.99, 0.001, 10000, 'models_logs_cnn_1', 'mnist_model_1'],
                       [100, 0.001, 0.95, 0.0001, 10000, 'models_logs_cnn_2', 'mnist_model_2'],
                       [100, 0.0005, 0.99, 0.003, 20000, 'models_logs_cnn_3', 'mnist_model_3'],
                       [100, 0.0001, 0.999, 0.001, 30000, 'models_logs_cnn_4', 'mnist_model_4']]
    pool.map_async(train_test, hyper_para_list)
    pool.close()
    pool.join()
    print('Done')


if __name__ == '__main__':
    tf.app.run()
