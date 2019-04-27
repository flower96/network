import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm


def conv(input, filter, kernel_size, stride, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel_size,
                             strides=stride, use_bias=use_bias, padding=padding)

        return x


def batch_normalization(inputs, training, scope_name):
    """
   function:
       batch normalization
    """
    with arg_scope([batch_norm],
                   scope=scope_name,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True,
                   reuse=tf.AUTO_REUSE):
        return batch_norm(inputs=inputs, is_training=training)


def flatten(x):
    return tf.layers.flatten(x)


def relu(x):
    return tf.nn.relu(x)


def avg_pooling(x):
    return tf.layers.average_pooling2d(inputs=x, pool_size=2, strides=2, padding='SAME')


def global_avg_pooling(x, strides=1):
    width = x.shape[1]
    height = x.shape[2]
    pool_size = [width, height]
    net = tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=strides)
    return net


def max_pooling(x):
    return tf.layers.max_pooling2d(inputs=x, pool_size=3, strides=2, padding='VALID')


def fully_conneted(input, class_num, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(input)
        x = tf.layers.dense(inputs=x, units=class_num, use_bias=use_bias)

        return x


# downsample为True表示两个res_block之间的操作
def res_block(input_x, channels, is_training=True, use_bias=True, downsample=False, scope='res_block'):
    with tf.variable_scope(scope):
        x = batch_normalization(input_x, is_training, scope_name='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv(input=shortcut, filter=channels, kernel_size=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_normalization(x, is_training, scope_name='batch_norm_3x3')
        x = relu(x)

        if downsample:
            x = conv(input=x, filter=channels, kernel_size=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel_size=1, stride=2, use_bias=use_bias, scope='conv_init')

        else:
            x = conv(input=x, filter=channels, kernel_size=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel_size=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_normalization(x, is_training, scope_name='batch_norm_1x1_back')
        x = relu(x)
        x = conv(input=x, filter=channels * 4, kernel_size=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        x = x + shortcut

        return x


ch = 64


class ResNet():
    def __init__(self, class_num=4):
        self.class_num = class_num

        self.inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])  # the input of the network
        self.labels = tf.placeholder(tf.float32, [None, self.class_num])  # the labels of train sampels
        self.training = tf.placeholder(tf.bool)

        self.prob = self.build(self.inputs)

        self.cost = -tf.reduce_mean(self.labels * tf.log(tf.clip_by_value(self.prob, 1e-10, 1.0)))

    def build(self, inputs, scope="ResNet50"):
        with tf.variable_scope(scope):

            inputs = inputs
            assert inputs.get_shape().as_list()[1:] == [224, 224, 3], 'the size of inputs is incorrect!'

            # x = tf.nn.conv2d(input=x, filter=64, kernel_size=[7,7], strides=2, padding='VALID', use_bias=False)

            x = conv(input=inputs, filter=ch, kernel_size=7, stride=2, padding='SAME', use_bias=False, scope='conv_first')
            print(x.shape)
            x = batch_normalization(x, self.training, scope_name='batch_norm_first')
            x = relu(x)
            x = max_pooling(x)
            print(x.shape)

            for i in range(3):
                x = res_block(x, channels=ch, is_training=self.training, downsample=False, scope='resblock0_' + str(i))
            print(x.shape)
            ########################################################################################################

            x = res_block(x, channels=ch * 2, is_training=self.training, downsample=True, scope='resblock1_0')
            print(x.shape)
            for i in range(1, 4):
                x = res_block(x, channels=ch * 2, is_training=self.training, downsample=False, scope='resblock1_' + str(i))
            print(x.shape)
            ########################################################################################################

            x = res_block(x, channels=ch * 4, is_training=self.training, downsample=True, scope='resblock2_0')
            print(x.shape)
            for i in range(1, 6):
                x = res_block(x, channels=ch * 4, is_training=self.training, downsample=False, scope='resblock2_' + str(i))
            print(x.shape)
            ########################################################################################################

            x = res_block(x, channels=ch * 8, is_training=self.training, downsample=True, scope='resblock_3_0')
            print(x.shape)
            for i in range(1, 3):
                x = res_block(x, channels=ch * 8, is_training=self.training, downsample=False, scope='resblock_3_' + str(i))
            print(x.shape)
            ########################################################################################################


            x = batch_normalization(x, self.training, scope_name='batch_norm_final')
            x = relu(x)
            print(x.shape)
            x = global_avg_pooling(x)
            print(x.shape)
            x = tf.layers.dense(x, units=ch)
            print(x.shape)
            x = fully_conneted(x, class_num=self.class_num, scope='logit')
            print(x.shape)
            x = tf.nn.softmax(x)
            return x


tf.reset_default_graph()
model = ResNet()

