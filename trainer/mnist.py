from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def inference(images, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1
  with tf.name_scope('hidden1'):
      with tf.name_scope("weights"):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        variable_summaries(weights)
      with tf.name_scope("biases"):
        biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
        variable_summaries(biases)

      with tf.name_scope("Wx_plus_b"):
          preactivate = tf.matmul(images, weights) + biases
          tf.summary.histogram('pre_activations', preactivate)

      hidden1 = tf.nn.relu(preactivate)
      tf.summary.histogram('activations', hidden1)

  # Hidden 2
  with tf.name_scope('hidden2'):
      with tf.name_scope("weights"):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        variable_summaries(weights)
      with tf.name_scope("biases"):
        biases = tf.Variable(tf.zeros([hidden2_units]),
                             name='biases')
        variable_summaries(biases)

      with tf.name_scope("Wx_plus_b"):
        preactivate = tf.matmul(hidden1, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)

      hidden2 = tf.nn.relu(preactivate)
      tf.summary.histogram('activations', hidden2)

  # Linear
  with tf.name_scope('softmax_linear'):
      with tf.name_scope("weights"):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        variable_summaries(weights)
      with tf.name_scope("biases"):
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        variable_summaries(biases)
      with tf.name_scope("Wx_plus_b"):
        logits = tf.matmul(hidden2, weights) + biases
        variable_summaries(logits)
  return logits

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  with tf.name_scope("cross_entropy"):
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  tf.summary.scalar('cross_entropy', cross_entropy)
  return cross_entropy

def training(loss, learning_rate):
    """
    Execute model fitting process here.

    :param loss: loss function
    :param learning_rate: rate of learning
    :return: train operation
    """
    with tf.name_scope("train"):
        # Create the gradient descent optimizer with the given learning rate.
        # TODO: Can update to pass as parameter
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op

def evaluation(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
          range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """

    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct = tf.nn.in_top_k(logits, labels, 1)
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_sum(tf.cast(correct, tf.int32))
    tf.summary.scalar('accuracy', accuracy)

    # Return the number of true entries.
    return accuracy



