# coding: utf-8

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import datetime
import math
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten


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
