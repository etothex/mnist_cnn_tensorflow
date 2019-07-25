#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# @Author  : GaoYQ
# @Time    : 2019/7/10 17:03
# @Software: PyCharm

import tensorflow as tf
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
num_gpu = 2

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
    parser.add_argument('--training_steps', default=20000, type=int)
    parser.add_argument('--buffer_size', default=10000, type=int)
    parser.add_argument('--count', help='dataset repeat count', default=100, type=int)
    parser.add_argument('--model_save_path', default='models_logs_cnn_multi_gpu')
    parser.add_argument('--model_name', default='mnist_cnn_multi_gpu')
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
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.ImageSize, self.ImageSize, self.ImageChannel],
                                name='x-input')
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, ], name='y-input')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        with tf.name_scope('single_model_inference'):
            self.logit = self.inference(self.x)
        # creat session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.summary_writer = None

    def __del__(self):
        # close session
        self.session.close()
        del self.session
        del self.logit
        del self.x
        del self.y
        del self.is_training
        self.summary_writer.close()
        del self.summary_writer

    def inference(self, inputs):
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
            conv1 = tf.nn.conv2d(inputs, conv1_weight, [1, 1, 1, 1], padding='SAME')
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

    def get_loss(self, samples, labels, l2_loss_rate, scope):
        logits = self.inference(samples)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        regularization_loss = l2_loss_rate * tf.add_n(tf.get_collection('l2_losses', scope))
        loss = cross_entropy_mean + regularization_loss
        return loss

    @staticmethod
    def average_gradients(tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def train(self,
              dataset,
              batch_size,
              learning_rate_base, learning_rate_decay,
              l2_loss_rate,
              moving_average_decay,
              training_steps,
              model_save_path, model_name, save_internal,
              validate_interval):
        # get batch
        image_batch, label_batch = dataset.train_batch(num_gpu * batch_size)
        # iteration
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # learning rate
        learning_rate = tf.train.exponential_decay(learning_rate_base,
                                                   global_step,
                                                   num_train_examples / batch_size,
                                                   learning_rate_decay,
                                                   staircase=True)
        tf.summary.scalar('learning-rate', learning_rate)
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # optimize using multiple GPUs
        all_losses = []
        tower_grads = []
        # print(tf.get_variable_scope().reuse)    # False
        # 定义一个variable_scope, 用当前variable_scope初始化, 便于共享模型参数.
        with tf.variable_scope(tf.get_variable_scope()), tf.name_scope('multi_gpu_model'):
            # split input to different gpu
            x_split = tf.split(value=self.x, num_or_size_splits=num_gpu, axis=0, name='split-x')
            y_split = tf.split(value=self.y, num_or_size_splits=num_gpu, axis=0, name='split-y')
            # Reuse variables in this scope(variable_scope not name_scope).
            # 初始化时已经定义了cnn的变量. 将控制变量重用的参数设为True, 这样不同的gpu会更新同一组参数.
            # print(tf.get_variable_scope().reuse)    # False
            tf.get_variable_scope().reuse_variables()
            # print(tf.get_variable_scope().reuse)    # True
            for i in range(num_gpu):
                with tf.device('/gpu:%d' % i):
                    # must use name_scope
                    with tf.name_scope('GPU_%d' % i) as scope:
                        current_loss = self.get_loss(x_split[i], y_split[i], l2_loss_rate, scope)
                        all_losses.append(current_loss)
                        grads = optimizer.compute_gradients(current_loss)
                        tower_grads.append(grads)
        # print(tf.get_variable_scope().reuse)    # False
        # calculate the mean of each gradient.
        # Note that this is the synchronization point across all towers.
        average_grad = self.average_gradients(tower_grads)
        average_loss = tf.reduce_mean(all_losses)
        tf.summary.scalar('loss-average', average_loss)
        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(average_grad, global_step=global_step)
        # moving average
        # ema = tf.train.ExponentialMovingAverage(decay=moving_average_decay, num_updates=global_step)
        # ema_op = ema.apply(tf.trainable_variables())
        # with tf.control_dependencies([apply_gradient_op, ema_op]):
        with tf.control_dependencies([apply_gradient_op]):
            train_op = tf.no_op(name='train')

        merge_op = tf.summary.merge_all()
        # creates a 'Saver', save graph and variables
        saver = tf.train.Saver()
        sess = self.session
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
                _, loss_mean, step, summary = sess.run([train_op, average_loss, global_step, merge_op],
                                                       feed_dict={self.x: images, self.y: labels, self.is_training: True})
            except tf.errors.OutOfRangeError as err:
                # print(err)
                print('dataset out of range ##### ', end='')
                print('step:', sess.run(global_step))
                break
            # save ckpt
            if i == 0 or (i+1) % save_internal == 0:
                saver.save(sess=sess,
                           save_path=os.path.join(model_save_path, model_name),
                           global_step=global_step)
                print("After %d training step(s), loss on training batch is %g." % (step, loss_mean))
                self.summary_writer.add_summary(summary=summary, global_step=step)
            # validation
            if i == 0 or (i+1) % validate_interval == 0:
                self.validation(dataset)

    def validation(self, dataset):
        image_batch, label_batch = dataset.eval_batch(dataset.dataset_val, 100)
        accuracy = self.evaluate(image_batch, label_batch)
        print('Validation accuracy:', accuracy)

    def test(self, dataset):
        image_batch, label_batch = dataset.eval_batch(dataset.dataset_test, 100)
        accuracy = self.evaluate(image_batch, label_batch)
        print('Test accuracy:', accuracy)

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


def main(argv=None):
    args = argument_parser(argv).parse_args()
    dataset = ReadDataset(ftrain=Data_path['train'],
                          fval=Data_path['validation'],
                          ftest=Data_path['test'],
                          buffer_size=args.buffer_size,
                          count=args.count)
    model = CNNModel()
    model.train(dataset=dataset,
                batch_size=args.batch_size,
                learning_rate_base=args.learning_rate_base,
                learning_rate_decay=args.learning_rate_decay,
                l2_loss_rate=args.l2_loss_rate,
                moving_average_decay=args.moving_average_decay,
                training_steps=args.training_steps,
                model_save_path=args.model_save_path,
                model_name=args.model_name,
                save_internal=args.save_interval,
                validate_interval=args.validate_interval)
    model.test(dataset=dataset)


if __name__ == '__main__':
    tf.app.run()
