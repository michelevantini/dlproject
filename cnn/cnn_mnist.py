# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import os
import time

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

DATA_DIR = "../norm_dataset"

IMG_SIZE = 1536
N_CHS = 3
N_CLASSES = 4
N_EPOCHS = 50
BATCH_SIZE = 32
QUEUE_SIZE = 2

TRAIN_FILE = "train.tfrecord"
TEST_FILE = "test.tfrecord"
VALIDATION_FILE = "validation.tfrecord"

TRAINING_SET_SIZE = 2560
TEST_SET_SIZE = 320
VALIDATION_SET_SIZE = 320

FEATURES_LIST = {"image/encoded": tf.FixedLenFeature([], tf.string),
                 "image/height": tf.FixedLenFeature([], tf.int64),
                 "image/width": tf.FixedLenFeature([], tf.int64),
                 "image/filename": tf.FixedLenFeature([], tf.string),
                 "image/class/label": tf.FixedLenFeature([], tf.int64), }

N_CPU = 12
config = tf.ConfigProto(device_count={"CPU": N_CPU},
                                inter_op_parallelism_threads=12,
                                intra_op_parallelism_threads=12)

#N_FILTERS = [4, 4, 4, 4, 4, 4]
N_FILTERS = [16, 32, 32, 64,64, 32]
CONV_SIZES = [[3,3,N_CHS,N_FILTERS[0]]
    , [3,3, N_FILTERS[0],N_FILTERS[1]]
    , [3,3, N_FILTERS[1],N_FILTERS[2]]
    , [3,3, N_FILTERS[2],N_FILTERS[3]]
    , [3,3, N_FILTERS[3],N_FILTERS[4]]]
#FC_SIZES = [4, 4]
FC_SIZES = [256, 128]

def deepnn(x):
    #with tf.name_scope('reshape'):
    #  x_image = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, N_CHS])

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(CONV_SIZES[0])
        b_conv1 = bias_variable([N_FILTERS[0]])
        h_conv1 = tf.nn.leaky_relu(conv2d(x, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_3x3(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(CONV_SIZES[1])
        b_conv2 = bias_variable([N_FILTERS[1]])
        h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(CONV_SIZES[2])
        b_conv3 = bias_variable([N_FILTERS[2]])
        h_conv3 = tf.nn.leaky_relu(conv2d(h_pool2, W_conv3) + b_conv3)

    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)

    with tf.name_scope('conv4'):
        W_conv4 = weight_variable(CONV_SIZES[3])
        b_conv4 = bias_variable([N_FILTERS[3]])
        h_conv4 = tf.nn.leaky_relu(conv2d(h_pool3, W_conv4) + b_conv4)

    with tf.name_scope('pool4'):
        h_pool4 = max_pool_3x3(h_conv4)

    with tf.name_scope('conv5'):
        W_conv5 = weight_variable(CONV_SIZES[4])
        b_conv5 = bias_variable([N_FILTERS[4]])
        h_conv5 = tf.nn.leaky_relu(conv2d(h_pool4, W_conv5) + b_conv5)

    with tf.name_scope('pool5'):
        h_pool5 = max_pool_3x3(h_conv5)

    s = h_pool5.get_shape().as_list()
    FLAT_IMAGE_SIZE = s[1]*s[2]*s[3]
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([FLAT_IMAGE_SIZE, FC_SIZES[0]])
        b_fc1 = bias_variable([FC_SIZES[0]])

        h_pool5_flat = tf.reshape(h_pool5, [BATCH_SIZE, FLAT_IMAGE_SIZE])

    #with tf.name_scope('dropout0'):
        #keep_prob0 = tf.placeholder(tf.float32)
        #h_pool5_flat_drop = tf.nn.dropout(h_pool5_flat, keep_prob0)

        h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

    #with tf.name_scope('dropout1'):
    #    keep_prob1 = tf.placeholder(tf.float32)
    #    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob1)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([FC_SIZES[0], FC_SIZES[1]])
        b_fc2 = bias_variable([FC_SIZES[1]])

        h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

    #with tf.name_scope('dropout2'):
    #    keep_prob2 = tf.placeholder(tf.float32)
    #    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob2)

    with tf.name_scope('fc3'):
        W_fc3 = weight_variable([FC_SIZES[1], N_CLASSES])
        b_fc3 = bias_variable([N_CLASSES])

        y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3
    return y_conv#, keep_prob0, keep_prob1, keep_prob2


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='SAME')


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features=FEATURES_LIST)
    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=N_CHS)
    image = tf.image.resize_image_with_crop_or_pad(image_raw, IMG_SIZE, IMG_SIZE)
    label = tf.cast(features["image/class/label"], tf.int64)
    return image, label

def inputs(dataset_file, batch_size):
    filename = os.path.join(DATA_DIR, dataset_file)

    with tf.name_scope("input" + dataset_file):
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.repeat(N_EPOCHS)
        dataset = dataset.map(decode)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def getOneHot(label_batch):
  label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
  return tf.one_hot(tf.add(label_batch, label_offset), depth=N_CLASSES, on_value=1.0, off_value=0.0)

def getOneHotTest(label_batch):
  label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="test_label_offset")
  return tf.one_hot(tf.add(label_batch, label_offset), depth=N_CLASSES, on_value=1.0, off_value=0.0)

def main(_):
  test_images, test_label = inputs(TEST_FILE, batch_size=BATCH_SIZE)
  test_label_one_hot = getOneHotTest(test_label)
  #x_test = tf.placeholder(tf.float32, [TEST_SET_SIZE, IMG_SIZE, IMG_SIZE, N_CHS])
  #y_test_ = tf.placeholder(tf.float32, [TEST_SET_SIZE, N_CLASSES])

  # Import data
  image_batch, label_batch = inputs(TRAIN_FILE, batch_size=BATCH_SIZE)
  label_one_hot = getOneHot(label_batch)
  # Create the model
  x = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_SIZE, IMG_SIZE, N_CHS])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [BATCH_SIZE, N_CLASSES])


  # Build the graph for the deep net
  #y_conv, keep_prob0, keep_prob1, keep_prob2 = deepnn(x)
  y_conv = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)
  regularisation_loss = tf.losses.get_regularization_loss()
  cross_entropy += regularisation_loss

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  y_pred = tf.placeholder(tf.float32, [BATCH_SIZE, N_CLASSES])
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    for j in range(N_EPOCHS):
        for i in range(int(TRAINING_SET_SIZE/BATCH_SIZE)):
          start_batch_time = time.time()
          image_out = sess.run(image_batch)
          label_one_hot_out = sess.run(label_one_hot)
          feed_dict = {x: image_out, y_: label_one_hot_out}#, keep_prob0: 0.45, keep_prob1: 0.45, keep_prob2: 0.45}
          train_step.run(feed_dict=feed_dict)

          infer_out, loss_out = sess.run([y_conv, cross_entropy], feed_dict=feed_dict)
          accuracy_out = accuracy.eval(feed_dict={y_:label_one_hot_out, y_pred:infer_out})

          #if i % 5 == 0:
          end_batch_time = time.time()
          batch_time = end_batch_time - start_batch_time
          print(i, " - Loss: ", loss_out, " - Accuracy: ", accuracy_out, " - Time: ", batch_time)

        print("Epoch ", j, " finished")
        print("Running now the model on test data")

        total_test_accuracy = 0.0
        for k in range(int(TEST_SET_SIZE/BATCH_SIZE)):
            test_image_out = sess.run(test_images)
            test_label_one_hot_out = sess.run(test_label_one_hot)
            feed_dict = {x: test_image_out, y_: test_label_one_hot_out}#, keep_prob0: 1.0, keep_prob1: 1.0, keep_prob2: 1.0}
            test_infer = sess.run(y_conv, feed_dict=feed_dict)
            test_accuracy = accuracy.eval(feed_dict={y_: test_label_one_hot_out, y_pred : test_infer})
            total_test_accuracy += test_accuracy*BATCH_SIZE
        print("Test accuracy: ", total_test_accuracy)
        print("-------------------------------------")
if __name__ == '__main__':
  tf.app.run(main=main)
