import tensorflow as tf


def loss_mse(prediction, target):
    loss = tf.math.reduce_mean(tf.square((prediction - target)))
    return loss


def loss_mse_adpwt(prediction, target, adpwt):
    loss = tf.math.reduce_mean(tf.square((prediction - target) * adpwt))
    return loss


def loss_inf(prediction, target):
    loss = tf.math.reduce_max(tf.square((prediction - target)))
    return loss


def loss_norm(prediction, target):
    loss = tf.reduce_mean(tf.norm(prediction - target, 2, axis=1) / tf.norm(target, 2, axis=1))
    return loss


def error_rel(prediction, target):
    error = tf.reduce_mean(tf.norm(prediction - target, 2, axis=(1)) / tf.norm(target, 2, axis=(1)), axis=0)
    # error = tf.reduce_mean(tf.square(prediction - target)/(tf.square(target)+ 1e-4))
    return error


def error_l2(prediction, target):
    error = tf.norm(prediction - target, ord='fro', axis=(0, 1))
    return error
