import tensorflow as tf
import numpy as np

def unit_dropout(w, params, is_training):
    if not is_training:
        return w

    w_shape = w.shape
    w = tf.reshape(w, [-1, w.shape[-1]])
    mask = tf.to_float(
        tf.random_uniform([int(w.shape[1])]) > params)[None, :]

    w = tf.reshape(mask * w, w_shape) # *为点乘

    return mask, w / (1 - params)

def untargeted_weight(w, params, is_training):
    if not is_training:
      return w
    return tf.nn.dropout(w, keep_prob=(1. - params))

def targeted_weight_dropout(w, drop_rate, targ_rate,is_training):
  drop_rate = drop_rate
  targ_perc = targ_rate

  w_shape = w.shape
  w = tf.reshape(w, [-1, w_shape[-1]])
  norm = tf.abs(w)
  idx = tf.to_int32(targ_perc * tf.to_float(tf.shape(w)[0]))
  threshold = tf.contrib.framework.sort(norm, axis=0)[idx]
  mask = norm < threshold[None, :]

  if not is_training:
    w = (1. - tf.to_float(mask)) * w
    w = tf.reshape(w, w_shape)
    return w

  mask = tf.to_float(
      tf.logical_and(tf.random_uniform(tf.shape(w)) < drop_rate, mask))
  w = (1. - mask) * w
  w = tf.reshape(w, w_shape)
  return w
