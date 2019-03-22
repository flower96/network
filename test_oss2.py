# coding: utf-8

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import datetime
import math
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten

import argparse


FLAGS = None


def read_img(dirpath):
    #label_list = os.listdir(dirpath)
    label_list = tf.gfile.ListDirectory(dirpath)# label_list表示路径下类别的总和，共有2个类别label，长度为12,要使用tf.gfile.ListDirectory函数
    imgs = []
    labels = []

    print( label_list)
    for i in range(len(label_list)):

        path_tmp = os.path.join(dirpath, label_list[i])  # 某一个类别的绝对路径
        for im in tf.gfile.ListDirectory(path_tmp):

            img_path = os.path.join(path_tmp, im)
            img = tf.gfile.FastGFile(img_path).read()
            img = tf.image.decode_jpeg(img)  #使用tf.gfile.FastGFile读取图片，img的type是一个tensor
            #print(type(img))

            imgs.append(img)
            label = np.zeros((1, 2), dtype=float)  # 创建一个1*2的一维矩阵
            label[0][i] = 1  # 将一维矩阵对应的那个元素设置为1，例如第一类的label则为[0,1,]
            if (len(labels) == 0):
                labels = label
            else:
                labels = np.r_[labels, label]
                # labels = np.concatenate((labels, label),axis=0)

    return imgs, labels



# 将混淆矩阵以csv格式输出存储
# 参数介绍：csv_path: 写入csv文件的路径； data_list:类别名称的列表；Output:混淆矩阵
def save_confusion_mat(csv_path, datalist, Output):
    n_classes = len(datalist)
    with open(csv_path, 'w') as f:
        f.write(", ")  # 逗号代表换格
        for key in range(n_classes):
            f.write("{0}, ".format(datalist[key]))  # 将每个类别名称写入csv作为行
        f.write("\n")

        for i in range(n_classes + 1):
            if i < n_classes:
                f.write("{0}, ".format(datalist[i]))  # 将每个类别名称写入csv作为列
            else:
                stri = 'total'
                f.write("{0}, ".format(stri))
            for j in range(n_classes + 2):
                f.write("{0}, ".format(Output[i][j]))
            f.write("\n")


# focal_loss，logits需要经过softmax操作后传入
def focal_loss(labels, logits, gamma=2.0, alpha=0.25):
    #     focal loss for multi-classification
    #     FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    #     FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
    #     Notice: logits is probability after softmax
    #     gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    #     d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)

    epsilon = 1.e-9
    #     labels = tf.to_int64(labels)
    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)

    model_out = tf.add(logits, epsilon)
    ce = tf.multiply(labels, -tf.log(model_out))
    weight = tf.multiply(labels, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_sum(fl, axis=1)
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    return reduced_fl


def conv_layer(input, filter, kernel_size, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        net = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel_size, strides=stride,
                               padding='SAME')
        return net


# BN层
# def Batch_Normalization(input):
#     net = slim.batch_norm(inputs=input, zero_debias_moving_mean="true", decay=0.9, epsilon=0.001, scale="true", center="true", updates_collections=None)
#     return net
def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


# 平均池化层
# def avg_pool(input, stride=2):
#     net = slim.avg_pool2d(inputs=input, kernel_size=2, stride=stride, padding="valid")
#     return net

def Average_Pooling(x, pool_size=(2, 2), strides=2, padding="valid"):
    net = tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=strides, padding=padding)
    return net


# 最大池化层
def Max_Pooling(x, pool_size=(3, 3), stride=2, padding="same"):
    net = tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
    return net


# Relu层
def Relu(input):
    net = tf.nn.relu(input)
    return net


# 全局池化层
def Global_Average_Pooling(x, stride=1):
    width = x.shape[1]
    height = x.shape[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)


def Linear(x):
    return tf.layers.dense(inputs=x, units=2, name='linear')


growth_k = 8
nb_block = 2
epsilon = 1e-4
learning_rate = 1e-6
batch_size = 16


class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        # [BN --> ReLU --> conv11 --> BN --> ReLU -->conv33]
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel_size=(1, 1), layer_name=scope + '_conv1')

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel_size=(3, 3), layer_name=scope + '_conv2')

            return x

    def transition_layer(self, x, scope):
        # [BN --> conv11 --> avg_pool2]
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel_size=(1, 1), layer_name=scope + '_conv1')
            x = Average_Pooling(x)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layer_concat = list()
            layer_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layer_concat.append(x)

            for i in range(nb_layers - 1):
                x = tf.concat(layer_concat, axis=3)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layer_concat.append(x)

            x = tf.concat(layer_concat, axis=3)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel_size=(7, 7), layer_name='conv0')
        print(x.shape)
        x = Max_Pooling(x, pool_size=[3, 3], stride=2)
        print(x.shape)

        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=24, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=16, layer_name='dense_final')

        x = Batch_Normalization(x, training=self.training, scope='final_batch')
        print(x.shape)
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)

        return x


def main(_):
    dirname = os.path.join(FLAGS.buckets, "")

    train_path = os.path.join(dirname, "train")
    test_path = os.path.join(dirname, "test")

    X_train, y_train = read_img(train_path)
    #X_val, y_val = read_img(test_path)

    x = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
    y = tf.placeholder(tf.float32, shape=[None, 2])
    training_flag = tf.placeholder(tf.bool)

    logits = DenseNet(x=x, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))


    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    train = optimizer.minimize(cost)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    total_batch = int(y_train.shape[0] / batch_size)
    print(total_batch)
    saver = tf.train.Saver(max_to_keep=1)
    config = tf.ConfigProto(allow_soft_placement=True)  # 自动选择可以运行的设备
    config.gpu_options.allow_growth = True  # Gpu内存按需增长

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model_dir = FLAGS.checkpointDir + 'output'  #输出目录

        X_train = sess.run(X_train)    #读取图片

        if not tf.gfile.Exists(model_dir):
            tf.gfile.MkDir(model_dir)
            #os.makedirs(model_dir)

        train_acc = 0.0
        train_loss = 0.0
        train_acc_list = list()
        train_acc_list.append(0)
        start = 0
        all_batch = 5 * total_batch  # 训练轮数 * 每一轮的batch数量

        print(all_batch)

        for i in range(1, all_batch + 1):

            if (i % 10 == 0):
                print("------------------第" + str(i) + "个batch----------------------")

            start = ((i - 1) * batch_size) % len(X_train)
            end = min(start + batch_size, len(X_train))

            x_batch = np.array(X_train[start: end])

            y_batch = y_train[start: end]

            train_feed_dict = {
                x: x_batch,
                y: y_batch,
                training_flag: True
            }

            _, batch_loss, batch_acc = sess.run([train, cost, accuracy], feed_dict=train_feed_dict)

            train_loss += batch_loss
            train_acc += batch_acc

            if (i % total_batch == 0):
                train_loss /= total_batch
                train_acc /= total_batch

                print(int(i / total_batch))
                print(train_acc_list)

                if train_acc > max(train_acc_list):
                    # model_path = os.path.join(model_dir , "Model_"+str(int(i/total_batch))+ "-epoch"+"_"+str(int(i/200))+ "-step")  # 模型的路径
                    model_path = os.path.join(model_dir, "model_" + str(int(i / total_batch)))
                    # save_confusion_mat(confusion_mat_path, datalist, Output)   #保存混淆矩阵
                    print("save successfully!")
                    saver.save(sess, model_path)  # 保存模型
                    train_acc_list.append(train_acc)


# 参数介绍：epoch：训练轮数；save_path：保存路径(模型路径和混淆矩阵路径都在该路径目录下)；batch_size：batch的大小
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 获得buckets路径
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    # 获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    #FLAGS.buckets = r"/notebooks/18_HCB/TBS/"
    #FLAGS.checkpointDir = r"/notebooks/18_HCB/TBS/out/"

    tf.app.run(main=main)