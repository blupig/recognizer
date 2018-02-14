# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def cnn_model_fn(features, labels, mode):
    """Model function"""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels].
    # Input images are 64x64 pixels, 3 channels (RGB).
    network = tf.reshape(features["x"], [-1, 64, 64, 3])

    # Convolutional Layer #1
    # Input shape: [batch_size, 64, 64, 3]
    # Output shape: [batch_size, 64, 64, 64]
    network = tf.layers.conv2d(inputs=network, filters=64, kernel_size=[16, 16], padding="same", activation=tf.nn.relu)

    # Pooling Layer #1
    # Input shape: [batch_size, 64, 64, 64]
    # Output shape: [batch_size, 32, 32, 64]
    network = tf.layers.max_pooling2d(inputs=network, pool_size=[2, 2], strides=2)

    # Input shape: [batch_size, 32, 32, 64]
    # Output shape: [batch_size, 32, 32, 32]
    network = tf.layers.conv2d(inputs=network, filters=32, kernel_size=[8, 8], padding="same", activation=tf.nn.relu)

    # Input shape: [batch_size, 32, 32, 32]
    # Output shape: [batch_size, 16, 16, 32]
    network = tf.layers.max_pooling2d(inputs=network, pool_size=[2, 2], strides=2)

    # Input shape: [batch_size, 16, 16, 32]
    # Output shape: [batch_size, 16, 16, 16]
    network = tf.layers.conv2d(inputs=network, filters=16, kernel_size=[4, 4], padding="same", activation=tf.nn.relu)

    # Input shape: [batch_size, 16, 16, 16]
    # Output shape: [batch_size, 8, 8, 16]
    network = tf.layers.max_pooling2d(inputs=network, pool_size=[2, 2], strides=2)

    # Flatten
    # Input Tensor Shape: [batch_size, 8, 8, 16]
    # Output Tensor Shape: [batch_size, 8 * 8 * 16]
    network = tf.reshape(network, [-1, 8 * 8 * 16])

    # Dense layer
    network = tf.layers.dense(inputs=network, units=512, activation=tf.nn.relu)

    # Dropout
    network = tf.layers.dropout(inputs=network, rate=0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))

    # Logits layer
    network = tf.layers.dense(inputs=network, units=10)

    # # Convolutional Layer #2 and Pooling Layer #2
    # conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    # logits = tf.layers.dense(inputs=dropout, units=10)

    # Generate predictions (EVAL and PREDICT)
    predictions = {
        "classes": tf.argmax(input=network, axis=1),
        "probabilities": tf.nn.softmax(network, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss (TRAIN and EVAL)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=network)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
