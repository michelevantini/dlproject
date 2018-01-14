"""Convolutional neural network to classify breast cancer images.

Some code borrowed from: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py

"""

import tensorflow as tf  # tensorflow module
import numpy as np  # numpy module
import os  # path join

from build_image_data import split_dataset
# from .build_image_data import split_dataset
import time
from vae import VariationalAutoencoder, xavier_init


# --- Constants ---
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, os.path.pardir, "norm_dataset")
TRAIN_FILE = "train.tfrecord"
TEST_FILE = "test.tfrecord"
VALIDATION_FILE = "validation.tfrecord"
# assert os.path.exists(os.path.join(DATA_DIR, TRAIN_FILE))
# assert os.path.exists(DATA_DIR, VALIDATION_FILE)

TRAINING_SET_SIZE = 2560
TEST_SET_SIZE = 320
VALIDATION_SET_SIZE = 320

FEATURES_LIST = {"image/encoded": tf.FixedLenFeature([], tf.string),
                 "image/height": tf.FixedLenFeature([], tf.int64),
                 "image/width": tf.FixedLenFeature([], tf.int64),
                 "image/filename": tf.FixedLenFeature([], tf.string),
                 "image/class/label": tf.FixedLenFeature([], tf.int64), }
# -----------------


# --- Settings ---
N_CPU = 12
SPLIT_DATASET = False
USING_TEST_SUITE = False
MAX_LEN_VALUES_LIST = 20

N_EPOCHS = 50
BATCH_SIZE = 32  # using a power of 2
QUEUE_SIZE = 2
IMAGE_SIZE = 512
N_CHANNELS = 3
N_CLASSES = 4
FILTERS_CHOICES_LAYER_1 = [2, 3, 5, 4, 6, 10, 7]
LEARNING_RATE = 0.001
CNN_REGULARIZATION = 0.0002
FC_REGULARIZATION = 0.0002
PRINT_RESOLUTION = 1  # print an update every x iterations

OPTIMIZER = tf.train.AdamOptimizer
NETWORK_CHOICE = "default"  # choices: "default", "original", "test_1", "simple_1"


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features=FEATURES_LIST)
    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=N_CHANNELS)
    # image_object = _image_object()
    image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    # image_object.image = image
    # image_object.height = features["image/height"]
    # image_object.width = features["image/width"]
    # image_object.filename = features["image/filename"]
    label = tf.cast(features["image/class/label"], tf.int64)
    # image_object.label = label
    return image, label


def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def inputs(dataset_file, batch_size, num_epochs):
    """Reads input data num_epochs times.

    Parameters
    ----------
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to train forever.

    Returns
    -------
        A tuple (images, labels), where:

        * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
        * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).

        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
    """
    if not num_epochs:
        num_epochs = None
    filename = os.path.join(DATA_DIR, dataset_file)

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

        #dataset = dataset.shuffle(QUEUE_SIZE * batch_size)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def test_inputs(dataset_file, batch_size, num_epochs):
    """Reads input data num_epochs times.

    Parameters
    ----------
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to train forever.

    Returns
    -------
        A tuple (images, labels), where:

        * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
        * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).

        This function creates a one_shot_iterator, meaning that it will only iterate
        over the dataset once. On the other hand there is no special initialization
        required.
    """
    if not num_epochs:
        num_epochs = None
    filename = os.path.join(DATA_DIR, dataset_file)

    with tf.name_scope('test_input'):
        # TFRecordDataset opens a protobuf and reads entries line by line
        # could also be [list, of, filenames]

        dataset = tf.data.TFRecordDataset(filename)

        dataset = dataset.repeat(num_epochs)

        # map takes a python function and applies it to every sample
        dataset = dataset.map(decode)

        # dataset = dataset.map(augment)
        # dataset = dataset.map(normalize)

        # the parameter is the queue size

        #dataset = dataset.shuffle(QUEUE_SIZE * batch_size)

        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def generate_cnn(image_batch, n_filters, filters_sizes, pool_sizes, reuse, flatten=True):

    # Check sizes are coherent.
    assert len(n_filters) == len(filters_sizes) == len(pool_sizes)

    with tf.variable_scope('ConvNet', reuse=reuse):
        conv_input = tf.reshape(image_batch, [-1, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS])
        pool = None
        after_pool_size = IMAGE_SIZE
        regularizer = tf.contrib.layers.l2_regularizer(scale=CNN_REGULARIZATION)
        for i in range(len(n_filters)):
            conv = tf.layers.conv2d(
                inputs=conv_input,
                filters=n_filters[i],
                kernel_size=filters_sizes[i],
                strides=(1, 1),
                activation=tf.nn.relu,
                kernel_regularizer=regularizer
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


        FINAL_SIZE = after_pool_size*after_pool_size*n_filters[-1]
        # Flatten for fully connected.
        pool_flat = tf.reshape(pool, [-1, FINAL_SIZE])

    if flatten:
        return pool_flat
    else:
        return pool


def generate_fc(fc_input, layer_sizes):
    """Creates a regularised fully connected network.

    Parameters
    ----------
    fc_input: Tensor input
        Input layer of the network.
    layer_sizes: list of int
        List of sizes for the layers of the network.

    Returns
    -------
    Output tensor.
        A tensorflow tensor representing the network graph.

    """

    for i in range(len(layer_sizes)):
        regularizer = tf.contrib.layers.l2_regularizer(scale=FC_REGULARIZATION)
        fc_input = tf.layers.dense(
            inputs=fc_input,
            units=layer_sizes[i],
            activation=tf.nn.relu,
            kernel_regularizer=regularizer
        )

    W = weight_variable([fc_input.get_shape().as_list()[-1], N_CLASSES])
    b = weight_variable([N_CLASSES])
    result = tf.nn.softmax(tf.matmul(fc_input, W) + b)

    return result


def train_fn(test=False,
             check_after_x_iterations=None,
             check_again_after_x=None,
             good_cross_entropy_threshold=None):
    """Function to run the train session."""
    with tf.Graph().as_default():
        image_batch, label_batch = inputs(TRAIN_FILE, batch_size=BATCH_SIZE,
                                          num_epochs=N_EPOCHS)

        image_batch_placeholder = tf.placeholder(tf.float32,
                                                 shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS])
        image_batch = tf.reshape(image_batch, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS))

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
                reuse=tf.AUTO_REUSE,
                flatten=True
            )

            network_fc = generate_fc(
                fc_input=network_cnn,
                layer_sizes=[256, 128]
            )

        elif NETWORK_CHOICE == "simple_1":
            network_cnn = generate_cnn(
                image_batch=image_batch_placeholder,
                n_filters=[8, 16, 32, 16],
                filters_sizes=[[2, 2], [2, 2], [3, 3], [3, 3]],
                pool_sizes=[[2, 2], [2, 2], [2, 2], [3, 3]],
                reuse=tf.AUTO_REUSE,
                flatten=True
            )

            network_fc = generate_fc(
                fc_input=network_cnn,
                layer_sizes=[]
            )

        # --- Network for testing configuration 1 ---
        elif NETWORK_CHOICE == "test_1":
            network_cnn = generate_cnn(
                image_batch=image_batch_placeholder,
                n_filters=[64, 32, 64, 32, 64, 32],
                filters_sizes=[[2, 2], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]],
                pool_sizes=[[3, 3], [3, 3], [2, 2], [2, 2], [3, 3], [3, 3]],
                reuse=tf.AUTO_REUSE,
                flatten=True
            )

            network_fc = generate_fc(
                fc_input=network_cnn,
                layer_sizes=[2048, 512, 256, 64, 16]
            )

        # --- Default network ---
        elif NETWORK_CHOICE == "default":
            # Create the network_cnn structure
            network_cnn = generate_cnn(
                image_batch=image_batch_placeholder,
                n_filters=[32],
                filters_sizes=[[3, 3]],
                pool_sizes=[[2, 2]],
                reuse=tf.AUTO_REUSE,
                flatten=True
            )

            # Add the fc at the end.
            network_fc = generate_fc(
                fc_input=network_cnn,
                layer_sizes=[]
            )
        else:
            raise ValueError("Nonexistent NETWORK_CHOICE: {}".format(NETWORK_CHOICE))


        loss = tf.losses.softmax_cross_entropy(onehot_labels=label_batch_placeholder,
                                               logits=network_fc)
        regularisation_loss = tf.losses.get_regularization_loss()
        loss += regularisation_loss

        infer_placeholder = tf.placeholder(tf.float32,
                                            shape=[BATCH_SIZE, N_CLASSES])
        label_placeholder = tf.placeholder(tf.float32,
                                            shape=[BATCH_SIZE, N_CLASSES])
        pred = tf.argmax(infer_placeholder, 1)
        label = tf.argmax(label_placeholder, 1)
        correct_prediction = tf.equal(pred, label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        train_step = OPTIMIZER().minimize(loss)
        # saver = tf.train.Saver()

        config = tf.ConfigProto(device_count={"CPU": N_CPU},
                                inter_op_parallelism_threads=N_CPU+2,
                                intra_op_parallelism_threads=N_CPU+2)

        with tf.Session(config=config) as sess:
            if not USING_TEST_SUITE:
                print("Starting training.. Using {} network architecture.".format(NETWORK_CHOICE))
            # Visualize the graph through tensorboard.
            # file_writer = tf.summary.FileWriter("./logs", sess.graph)

            sess.run(tf.global_variables_initializer())
            # saver.restore(sess, "dltmp/checkpoint-train.ckpt")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            # Start taking time (full)
            start_time = time.time()
            step = 0
            check_iteration = 0  # as step but may be resetted
            last_values_list = []
            try:
                while True:
                    image_out, label_out, label_batch_one_hot_out = sess.run(
                        [image_batch, label_batch, label_batch_one_hot])

                    # Start taking time (iteration)
                    start_time_iteration = time.time()
                    _, infer_out, loss_out = sess.run([train_step, network_fc, loss],
                                                      feed_dict={image_batch_placeholder: image_out,
                                                                 label_batch_placeholder: label_batch_one_hot_out})
                    #sess.run(tf.round(infer_out))

                    # Save time (iteration)
                    iteration_time = time.time() - start_time_iteration

                    if step % PRINT_RESOLUTION == 0:

                        # Check average improvement
                        last_values_list.append(loss_out)
                        # Don't make the list too large
                        if len(last_values_list) > MAX_LEN_VALUES_LIST:
                            last_values_list = last_values_list[1:]

                        if step > 0:
                            # Check average improvement
                            half = len(last_values_list) // 2
                            first_half_list = last_values_list[:half]
                            second_half_list = last_values_list[half:]

                            imrovement_str_list = ["BAD", "UNCLEAR", "LOW", "PROMISING", "GOOD"]
                            if np.average(second_half_list) < np.min(first_half_list):
                                improvement_grade = 4
                            elif np.average(second_half_list) < np.average(first_half_list):
                                improvement_grade = 3
                            elif np.min(second_half_list) < np.average(first_half_list):
                                improvement_grade = 2
                            elif np.min(second_half_list) < np.max(first_half_list):
                                improvement_grade = 1
                            else:
                                improvement_grade = 0

                            improvement_str = imrovement_str_list[improvement_grade]
                        else:
                            improvement_str = "Not available"
                            improvement_grade = 5

                        print("\t",
                              step, ": ", "loss: %.3f" % loss_out,
                              " | iteration time: %.2fs" % iteration_time,
                              " | accuracy: %.2f" % sess.run(accuracy, feed_dict={
                                  infer_placeholder: infer_out,
                                  label_placeholder: label_batch_one_hot_out
                              }),
                              " | improvement: ", improvement_str,
                              sep="")
                        #print(sess.run(tf.argmax(label_batch_one_hot, 1)))
                        #print(sess.run(tf.argmax(infer_out, 1)))

                        # Check if it's time to leave.
                        if USING_TEST_SUITE and \
                            step >= check_after_x_iterations and \
                            check_iteration >= check_again_after_x:
                            check_iteration = 0
                            if improvement_grade >= 3:
                                if loss_out < good_cross_entropy_threshold:
                                    good_cross_entropy_threshold = loss_out
                            elif loss_out < good_cross_entropy_threshold:
                                good_cross_entropy_threshold = loss_out
                            else:
                                raise NoImprovementsException("", "")

                    # if(i%50 == 0):
                    #    saver.save(sess, "dltmp/checkpoint-train.ckpt")
                    step += 1
                    check_iteration += 1
            except tf.errors.OutOfRangeError:
                # Save time (full)
                total_running_time = time.time() - start_time
                print('Done training for %d epochs, %d steps, %1.2f seconds.' % (N_EPOCHS, step, total_running_time))

            if test:
                test_fn(network_fc, loss, image_batch_placeholder, label_batch_placeholder
                        , accuracy, infer_placeholder, label_placeholder)
                    #print(sess.run(tf.argmax(label_batch_one_hot, 1)))
                    #print(sess.run(tf.argmax(infer_out, 1)))
                    #print("Accuracy for this mini-batch: ", sess.run(accuracy))
                    #old_loss = loss_out

                #print(sess.run(tf.argmax(label_batch_one_hot, 1)))
                #print(sess.run(tf.argmax(infer_out, 1)))


            coord.request_stop()
            coord.join(threads)
            sess.close()


def test_fn(network_fc, loss, image_batch_placeholder, label_batch_placeholder
            , accuracy, infer_placeholder, label_placeholder):
    config = tf.ConfigProto(device_count={"CPU": N_CPU},
                            inter_op_parallelism_threads=N_CPU,
                            intra_op_parallelism_threads=N_CPU)

    with tf.Session(config=config) as sess:
        print("\tStarting testing.. Using {} network architecture.".format(NETWORK_CHOICE))
        # Visualize the graph through tensorboard.
        # file_writer = tf.summary.FileWriter("./logs", sess.graph)

        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, "dltmp/checkpoint-train.ckpt")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        test_images, test_label = test_inputs(TEST_FILE, batch_size=BATCH_SIZE, num_epochs=1)

        test_images_placeholder = tf.placeholder(tf.float32,
                                                 shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS])
        test_label_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_CLASSES])

        step = 0
        print("\t----------------")
        print("\tTesting started")
        test_accuracy = 0
        try:
            while True:
                test_label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="test_label_offset")
                test_label_one_hot = tf.one_hot(tf.add(test_label, test_label_offset),
                                                depth=N_CLASSES, on_value=1.0, off_value=0.0)

                test_images_out, test_label_out, test_label_one_hot_out = sess.run(
                    [test_images, test_label, test_label_one_hot])
                start_time_iteration = time.time()
                test_infer = sess.run(network_fc,
                                                 feed_dict={image_batch_placeholder: test_images_out,
                                                            label_batch_placeholder: test_label_one_hot_out})
                test_accuracy += BATCH_SIZE * sess.run(accuracy, feed_dict={infer_placeholder : test_infer
                                                            , label_placeholder : test_label_one_hot_out})
                # sess.run(tf.round(infer_out))
                # correct_prediction = tf.equal(tf.argmax(infer_out, 1), tf.argmax(label_batch_one_hot, 1))
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                # Save time (iteration)

                step += 1
        except tf.errors.OutOfRangeError:
            # Save time (full)
            print('\tTraining for %d epochs done, %d steps' % (N_EPOCHS, step))
        print("\tTest accuracy: %.3f " % (float(test_accuracy)/TEST_SET_SIZE))


def test_suite():
    optimizers_list = [tf.train.AdamOptimizer,
                       tf.train.AdadeltaOptimizer,
                       tf.train.MomentumOptimizer,
                       tf.train.GradientDescentOptimizer]

    learning_rates_list = [a * 10**(-exp) for exp in range(2, 7) for a in [1, 0.5]]

    minibatch_sizes_list = [32, 50, 64, 128]

    architechtures_list = ["simple_1", "test_1", "original", "default"]

    check_after_x_iterations = 600

    check_again_after_x = 100

    good_cross_entropy_threshold = 1.4

    print("Running test suite.")

    global NETWORK_CHOICE
    global OPTIMIZER
    global LEARNING_RATE
    global BATCH_SIZE
    global USING_TEST_SUITE
    USING_TEST_SUITE = True

    test_number = 0
    for arch in architechtures_list:
        for optimizer in optimizers_list:
            for minibatch_size in minibatch_sizes_list:
                for learning_rate in learning_rates_list:
                    NETWORK_CHOICE = arch
                    OPTIMIZER = optimizer
                    BATCH_SIZE = minibatch_size
                    LEARNING_RATE = learning_rate
                    print("Test: {} | network: {} | optimizer: {} | minibatch: {} | learning rate: {} |".format(
                        test_number, arch, OPTIMIZER.__name__, minibatch_size, learning_rate
                    ))
                    try:
                        train_fn(test=True,
                                 check_after_x_iterations=check_after_x_iterations,
                                 check_again_after_x=check_again_after_x,
                                 good_cross_entropy_threshold=good_cross_entropy_threshold)
                    except NoImprovementsException as no_improvement:
                        print("Stopping {}, no more improvement.\n\n".format(test_number))
                    test_number += 1


def train_vae(network_architecture, learning_rate=0.001, display_step=5):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=BATCH_SIZE)
    image_batch, label_batch = inputs(TRAIN_FILE, batch_size=BATCH_SIZE,
                                      num_epochs=N_EPOCHS)
    image_batch_placeholder = tf.placeholder(tf.float32,
                                             shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS])
    image_batch = tf.reshape(image_batch, (BATCH_SIZE, IMAGE_SIZE*IMAGE_SIZE*N_CHANNELS))
    # Training cycle
    for epoch in range(N_EPOCHS):
        avg_cost = 0.
        total_batch = int(TRAINING_SET_SIZE / BATCH_SIZE)
        # Loop over all batches
        for i in range(total_batch):
            print("...")
            batch_xs = vae.sess.run(image_batch)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / TRAINING_SET_SIZE * BATCH_SIZE

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae


network_architecture = \
    dict(n_hidden_recog_1=500, # 1st layer encoder neurons
         n_hidden_recog_2=500, # 2nd layer encoder neurons
         n_hidden_gener_1=500, # 1st layer decoder neurons
         n_hidden_gener_2=500, # 2nd layer decoder neurons
         n_input=IMAGE_SIZE*IMAGE_SIZE*3, # MNIST data input (img shape: 28*28)
         n_z=20)  # dimensionality of latent space


class NoImprovementsException(Exception):
    def __init__(self, message, errors):
        # Call the base class constructor with the parameters it needs
        super(NoImprovementsException, self).__init__(message)
        self.errors = errors


if __name__ == '__main__':
    if SPLIT_DATASET:
        with tf.Session() as sess:
            sess.run(split_dataset())
            sess.close()
    # train_fn(test=True)
    # vae = train_vae(network_architecture)
    # train_fn(test=True)
    # test_fn()
    test_suite()
