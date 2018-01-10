"""Convolutional neural network to classify breast cancer images.

Some code borrowed from: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py

"""

import tensorflow as tf  # tensorflow module
import numpy as np  # numpy module
import os  # path join
import time

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, os.path.pardir, "dataset")
TRAIN_FILE = "train.tfrecord"
VALIDATION_FILE = "validation.tfrecord"
assert os.path.exists(os.path.join(DATA_DIR, TRAIN_FILE))
# assert os.path.exists(DATA_DIR, VALIDATION_FILE)
TRAINING_SET_SIZE = 3200
N_EPOCHS = 5
BATCH_SIZE = 10
IMAGE_SIZE = 1536
N_CHANNEL = 3
N_CLASSES = 4
FILTERS_CHOICES_LAYER_1 = [2, 3, 5, 4, 6, 10, 7]
LEARNING_RATE = 0.000001
NETWORK_CHOICE = "test_1"  # choices: "default", "original", "test_1"

FEATURES_LIST = {"image/encoded": tf.FixedLenFeature([], tf.string),
                 "image/height": tf.FixedLenFeature([], tf.int64),
                 "image/width": tf.FixedLenFeature([], tf.int64),
                 "image/filename": tf.FixedLenFeature([], tf.string),
                 "image/class/label": tf.FixedLenFeature([], tf.int64), }


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features=FEATURES_LIST)
    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=N_CHANNEL)
    # image_object = _image_object()
    image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    # image_object.image = image
    # image_object.height = features["image/height"]
    # image_object.width = features["image/width"]
    # image_object.filename = features["image/filename"]
    label = tf.cast(features["image/class/label"], tf.int64)
    # image_object.label = label
    return image, label


def inputs(train, batch_size, num_epochs):
    """Reads input data num_epochs times.
    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.
    Returns:
        A tuple (images, labels), where:

        * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
        * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).

        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
    """
    if not num_epochs: num_epochs = None
    filename = os.path.join(DATA_DIR,
                            TRAIN_FILE if train else VALIDATION_FILE)

    with tf.name_scope('input'):
        # TFRecordDataset opens a protobuf and reads entries line by line
        # could also be [list, of, filenames]
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.repeat(num_epochs)

        # map takes a python function and applies it to every sample
        dataset = dataset.map(decode)
        # dataset = dataset.map(augment)
        # dataset = dataset.map(normalize)

        # the parameter is the queue size
        dataset = dataset.shuffle(2 * batch_size)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.02, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def generate_cnn(image_batch, n_filters, filters_sizes, pool_sizes, reuse, flatten=True):

    # Check sizes are coherent.
    assert len(n_filters) == len(filters_sizes) == len(pool_sizes)

    with tf.variable_scope('ConvNet', reuse=reuse):
        conv_input = tf.reshape(image_batch, [-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANNEL])
        pool = None
        after_pool_size = IMAGE_SIZE
        for i in range(len(n_filters)):
            conv = tf.layers.conv2d(
                inputs=conv_input,
                filters=n_filters[i],
                kernel_size=filters_sizes[i],
                strides=(1, 1),
                activation=tf.nn.relu
            )

            # Compute size of following layer.
            after_conv_size = int(after_pool_size - filters_sizes[i][0] + 1)

            pool = tf.layers.max_pooling2d(
                inputs=conv,
                pool_size=pool_sizes[i],
                strides=pool_sizes[i][0]
            )

            # Compute the size of the following layer.
            after_pool_size = int((after_conv_size - pool_sizes[i][0])/pool_sizes[i][0] + 1)

            # Output is the input of the next iteration.
            conv_input = pool

        # Flatten for fully connected.
        pool_flat = tf.reshape(pool, [-1, after_pool_size * after_pool_size * n_filters[-1]])

    if flatten:
        return pool_flat
    else:
        return pool


def generate_fc(fc_input, layer_sizes):
    for i in range(len(layer_sizes)):
        fc_input = tf.layers.dense(
            inputs=fc_input,
            units=layer_sizes[i]
        )

    return fc_input


def train_fn():
    with tf.Graph().as_default():
        image_batch, label_batch = inputs(train=True, batch_size=BATCH_SIZE,
                                          num_epochs=N_EPOCHS)

        image_batch_placeholder = tf.placeholder(tf.float32,
                                                 shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNEL])
        image_batch = tf.reshape(image_batch, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNEL))

        label_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_CLASSES])
        label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
        label_batch_one_hot = tf.one_hot(tf.add(label_batch, label_offset),
                                         depth=N_CLASSES, on_value=1.0, off_value=0.0)

        # *--------------------------*
        # | Create the network graph |
        # *--------------------------*
        # If you want to create another network graph,
        # give it a name and add it here under an *if* branch.
        global NETWORK_CHOICE  # use global scope, not local
        # --- Network following original paper ---
        if NETWORK_CHOICE == "original":
            network_cnn = generate_cnn(
                image_batch=image_batch_placeholder,
                n_filters=[16, 32, 64, 64, 32],
                filters_sizes=[[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                pool_sizes=[[3, 3], [2, 2], [2, 2], [3, 3], [3, 3]],
                reuse=False,
                flatten=True
            )

            network_fc = generate_fc(
                fc_input=network_cnn,
                layer_sizes=[256, 128, N_CLASSES]
            )

        # --- Network for testing configuration 1 ---
        if NETWORK_CHOICE == "test_1":
            network_cnn = generate_cnn(
                image_batch=image_batch_placeholder,
                n_filters=[64, 32, 64, 32, 64, 32],
                filters_sizes=[[2, 2], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                pool_sizes=[[3, 3], [3, 3], [2, 2], [2, 2], [3, 3], [3, 3]],
                reuse=False,
                flatten=True
            )

            network_fc = generate_fc(
                fc_input=network_cnn,
                layer_sizes=[512, 256, 128, 64, 16, N_CLASSES]
            )

        # --- Default network ---
        else:
            NETWORK_CHOICE = "default"
            # Create the network_cnn structure
            network_cnn = generate_cnn(
                image_batch=image_batch_placeholder,
                n_filters=[32],
                filters_sizes=[[3, 3]],
                pool_sizes=[[2, 2]],
                reuse=False,
                flatten=True
            )

            # Add the fc at the end.
            network_fc = generate_fc(
                fc_input=network_cnn,
                layer_sizes=[N_CLASSES]
            )

        # loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_one_hot, network_cnn=network_cnn))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_placeholder, logits=network_fc))

        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

        # saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Starting training.. Using {} network architecture.".format(NETWORK_CHOICE))
            # Visualize the graph through tensorboard.
            # file_writer = tf.summary.FileWriter("./logs", sess.graph)

            sess.run(tf.global_variables_initializer())
            # saver.restore(sess, "dltmp/checkpoint-train.ckpt")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            step = 0

            # Start taking time (full)
            start_time = time.time()

            try:
                while True:

                    image_out, label_out, label_batch_one_hot_out = sess.run(
                        [image_batch, label_batch, label_batch_one_hot])

                    # Start taking time (iteration)
                    start_time_iteration = time.time()
                    _, infer_out, loss_out = sess.run([train_step, network_fc, loss],
                                                      feed_dict={image_batch_placeholder: image_out,
                                                                 label_batch_placeholder: label_batch_one_hot_out})
                    # Save time (iteration)
                    iteration_time = time.time() - start_time_iteration

                    if step % 10 == 0:
                        print(
                            step, ": ", "loss: %.3f " % loss_out,
                            " - iteration time: %.2fs" % iteration_time,
                            sep="")
                        # print(image_out.shape)
                        # print("label_out: ")
                        # print(filename_out)
                        # print(label_out)
                        # print(label_batch_one_hot_out)
                        # print("infer_out: ")
                        # print(infer_out)
                    # if(i%50 == 0):
                    #    saver.save(sess, "dltmp/checkpoint-train.ckpt")
                    step += 1
            except tf.errors.OutOfRangeError:
                # Save time (full)
                total_running_time = time.time() - start_time
                print('Done training for %d epochs, %d steps, %1.2f seconds.' % (N_EPOCHS, step, total_running_time))

            coord.request_stop()
            coord.join(threads)
            sess.close()


if __name__ == '__main__':
    train_fn()
