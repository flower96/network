import tensorflow as tf
import numpy as np


def cbam_block(input_feature, name, ratio=0.5):
    #   """
    #   Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    #   As described in https://arxiv.org/abs/1807.06521.
    #   """
    with tf.variable_scope(name):
        attention_feature = channel_attention(input_feature, 'ch_at', ratio=0.5)
        attention_feature = spatial_attention(attention_feature, 'sp_at')
        print("CBAM Hello")
    return attention_feature


def channel_attention(input_feature, name, ratio=0.5):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape().as_list()[-1]

        #print(channel)

        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=int(channel * ratio),
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, int(channel * ratio))
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=int(channel * ratio),
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, int(channel * ratio))
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale


def spatial_attention(input_feature, name):
    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat


def cbam_module(inputs, reduction_ratio=0.5, name=""):
    with tf.variable_scope("cbam_" + name, reuse=tf.AUTO_REUSE):
        # 假如输入是[batsize,h,w,channel]，
        # channel attension 因为要得到batsize * 1 * 1 * channel，它的全连接层第一层
        # 隐藏层单元个数是channel / r, 第二层是channel，所以这里把channel赋值给hidden_num
        batch_size, hidden_num = inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[3]

        # 通道attension
        # 全局最大池化，窗口大小为h * w，所以对于这个数据[batsize,h,w,channel]，他其实是求每个h * w面积的最大值
        # 这里实现是先对h这个维度求最大值，然后对w这个维度求最大值，平均池化也一样
        maxpool_channel = tf.reduce_max(tf.reduce_max(inputs, axis=1, keepdims=True), axis=2, keepdims=True)
        avgpool_channel = tf.reduce_mean(tf.reduce_mean(inputs, axis=1, keepdims=True), axis=2, keepdims=True)

        # 上面全局池化结果为batsize * 1 * 1 * channel，它这个拉平输入到全连接层
        # 这个拉平，它会保留batsize，所以结果是[batsize,channel]
        maxpool_channel = tf.layers.Flatten()(maxpool_channel)
        avgpool_channel = tf.layers.Flatten()(avgpool_channel)

        # 将上面拉平后结果输入到全连接层，第一个全连接层hiddensize = channel/r = channel * reduction_ratio，
        # 第二哥全连接层hiddensize = channel
        mlp_1_max = tf.layers.dense(inputs=maxpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
                                    reuse=None, activation=tf.nn.relu)
        mlp_2_max = tf.layers.dense(inputs=mlp_1_max, units=hidden_num, name="mlp_2", reuse=None)
        # 全连接层输出结果为[batsize,channel]，这里又降它转回到原来维度batsize * 1 * 1 * channel，
        mlp_2_max = tf.reshape(mlp_2_max, [batch_size, 1, 1, hidden_num])

        mlp_1_avg = tf.layers.dense(inputs=avgpool_channel, units=int(hidden_num * reduction_ratio), name="mlp_1",
                                    reuse=True, activation=tf.nn.relu)
        mlp_2_avg = tf.layers.dense(inputs=mlp_1_avg, units=hidden_num, name="mlp_2", reuse=True)
        mlp_2_avg = tf.reshape(mlp_2_avg, [batch_size, 1, 1, hidden_num])

        # 将平均和最大池化的结果维度都是[batch_size,1,1,channel]相加，然后进行sigmod，维度不变
        channel_attention = tf.nn.sigmoid(mlp_2_max + mlp_2_avg)
        # 和最开始的inputs相乘，相当于[batch_size,1,1,channel] * [batch_size,h,w,channel]
        # 只有维度一样才能相乘,这里相乘相当于给每个通道作用了不同的权重
        channel_refined_feature = inputs * channel_attention

        # 空间attension
        # 上面得到的结果维度依然是[batch_size,h,w,channel]，
        # 下面要进行全局通道池化，其实就是一条通道里面那个通道的值最大，其实就是对channel这个维度求最大值
        # 每个通道池化相当于将通道压缩到了1维，有两个池化，结果为两个[batch_size,h,w,1]feature map
        maxpool_spatial = tf.reduce_max(inputs, axis=3, keepdims=True)
        avgpool_spatial = tf.reduce_mean(inputs, axis=3, keepdims=True)

        # 将两个[batch_size,h,w,1]的feature map进行通道合并得到[batch_size,h,w,2]的feature map
        max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=3)

        # 然后对上面的feature map用1个7*7的卷积核进行卷积得到[batch_size,h,w,1]的feature map，因为是用一个卷积核卷的
        # 所以将2个输入通道压缩到了1个输出通道
        conv_layer = tf.layers.conv2d(inputs=max_avg_pool_spatial, filters=1, kernel_size=(7, 7), padding="same",
                                      activation=None)
        # 然后再对上面得到的[batch_size,h,w,1]feature map进行sigmod，这里为什么要用一个卷积核压缩到1个通道，相当于只得到了一个面积的值
        # 然后进行sigmod，因为我们要求的就是feature map面积上不同位置像素的中重要性，所以它压缩到了一个通道，然后求sigmod
        spatial_attention = tf.nn.sigmoid(conv_layer)

        # 上面得到了空间attension feature map [batch_size,h,w,1]，然后再用这个和经过空间attension作用的结果相乘得到最终的结果
        # 这个结果就是经过通道和空间attension共同作用的结果
        refined_feature = channel_refined_feature * spatial_attention

    return refined_feature


def se_block(input_feature, name, ratio=0.5):
  # """Contains the implementation of Squeeze-and-Excitation(SE) block.
  # As described in https://arxiv.org/abs/1709.01507.
  # """
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape().as_list()[-1]

        # Global average pooling
        squeeze = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)
        assert squeeze.get_shape()[1:] == (1,1,channel)
        excitation = tf.layers.dense(inputs=squeeze,
                                 units=int(channel*ratio),
                                 activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='bottleneck_fc')

        assert excitation.get_shape()[1:] == (1,1,int(channel*ratio))
        excitation = tf.layers.dense(inputs=excitation,
                                 units=channel,
                                 activation=tf.nn.sigmoid,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='recover_fc')
        assert excitation.get_shape()[1:] == (1,1,channel)
        scale = input_feature * excitation
    return scale



def Global_Average_Pooling(x, strides=1):
    width = x.shape[1]
    height = x.shape[2]
    pool_size = [width, height]
    net = tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=strides)
    return net

def Fully_Connected(x, units, layer_name='fully_connected'):
    with tf.name_scope(layer_name):
        net = tf.layers.dense(inputs=x, use_bias=False, units=units)
        return net

def squeeze_excitation_layer(input_x, ratio, layer_name):
    # SE模块
    with tf.name_scope(layer_name):
        channel = input_x.get_shape().as_list()[-1]
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_Connected(squeeze, units=channel / ratio, layer_name=layer_name + '_fully_connected1')
        excitation = tf.nn.relu(excitation)
        excitation = Fully_Connected(excitation, units=channel, layer_name=layer_name + '_fully_connected2')
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, channel])
        scale = input_x * excitation

        return scale