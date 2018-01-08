import tensorflow as tf # tensorflow module
import numpy as np # numpy module
import os # path join


DATA_DIR = "../dataset/"
TRAINING_SET_SIZE = 1000
BATCH_SIZE = 50
IMAGE_SIZE = 1536
N_CHANNEL = 3
N_CLASSES = 4
FLAT_LEN = IMAGE_SIZE**2

FEATURES_LIST = {"image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),}



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# image object from protobuf
class _image_object:
    def __init__(self):
        self.image = tf.Variable([], dtype = tf.string)
        self.height = tf.Variable([], dtype = tf.int64)
        self.width = tf.Variable([], dtype = tf.int64)
        self.filename = tf.Variable([], dtype = tf.string)
        self.label = tf.Variable([], dtype = tf.int32)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features=FEATURES_LIST)
    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=N_CHANNEL)
    image_object = _image_object()
    image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    image_object.height = features["image/height"]
    image_object.width = features["image/width"]
    image_object.filename = features["image/filename"]
    image_object.label = tf.cast(features["image/class/label"], tf.int64)
    return image_object

def input_fn(filenames, if_random = True):
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)
    filename_queue = tf.train.string_input_producer(filenames)
    image_object = read_and_decode(filename_queue)
    image = tf.image.per_image_standardization(image_object.image)
#    image = image_object.image
#    image = tf.image.adjust_gamma(tf.cast(image_object.image, tf.float32), gamma=1, gain=1) # Scale image to (0, 1)
    label = image_object.label
    filename = image_object.filename

    if(if_random):
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
        print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
        num_preprocess_threads = 1
        image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue = min_queue_examples)
        return image_batch, label_batch, filename_batch
    else:
        image_batch, label_batch, filename_batch = tf.train.batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = 1)
        return image_batch, label_batch, filename_batch


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

def inference_fn(image_batch):
    #x = tf.placeholder(tf.float32, [None, flat_len])
    x_image = tf.reshape(image_batch, [-1, 1536*1536*3])
    W = tf.Variable(tf.zeros([1536*1536*3, 4]))
    b = tf.Variable(tf.zeros([4]))

    y = tf.nn.softmax(tf.matmul(x_image, W) + b)

    return y


def train_fn():
    image_batch_out, label_batch_out, filename_batch = input_fn([DATA_DIR + "train.tfrecord"])

    image_batch_placeholder = tf.placeholder(tf.float32
                            , shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNEL])
    image_batch = tf.reshape(image_batch_out
                            , (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNEL))

    label_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_CLASSES])
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    label_batch_one_hot = tf.one_hot(tf.add(label_batch_out, label_offset)
                                     , depth=N_CLASSES, on_value=1.0, off_value=0.0)

    logits_out = inference_fn(image_batch_placeholder)
#    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_one_hot, logits=logits_out))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_placeholder, logits=logits_out))

    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

    #saver = tf.train.Saver()

    with tf.Session() as sess:
        # Visualize the graph through tensorboard.
        file_writer = tf.summary.FileWriter("./logs", sess.graph)

        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, "dltmp/checkpoint-train.ckpt")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        for i in range(500):
            image_out, label_out, label_batch_one_hot_out, filename_out = sess.run([image_batch, label_batch_out, label_batch_one_hot, filename_batch])

            _, infer_out, loss_out = sess.run([train_step, logits_out, loss], feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})

            if i%10 == 0:
                print(i)
                print(image_out.shape)
                print("label_out: ")
                print(filename_out)
                print(label_out)
                print(label_batch_one_hot_out)
                print("infer_out: ")
                print(infer_out)
                print("loss: ")
                print(loss_out)
            #if(i%50 == 0):
            #    saver.save(sess, "dltmp/checkpoint-train.ckpt")

        coord.request_stop()
        coord.join(threads)
        sess.close()


'''
def flower_eval():
    image_batch_out, label_batch_out, filename_batch = flower_input(if_random = False, if_training = False)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 224, 224, 3])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_tensor_placeholder = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    label_batch = tf.add(label_batch_out, label_offset)

    logits_out = tf.reshape(flower_inference(image_batch_placeholder), [BATCH_SIZE, 5])
    logits_batch = tf.to_int64(tf.arg_max(logits_out, dimension = 1))

    correct_prediction = tf.equal(logits_batch, label_tensor_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "/home/dltmp/checkpoint-train.ckpt")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = sess)

        accuracy_accu = 0

        for i in range(29):
            image_out, label_out, filename_out = sess.run([image_batch, label_batch, filename_batch])

            accuracy_out, logits_batch_out = sess.run([accuracy, logits_batch], feed_dict={image_batch_placeholder: image_out, label_tensor_placeholder: label_out})
            accuracy_accu += accuracy_out

            print(i)
            print(image_out.shape)
            print("label_out: ")
            print(filename_out)
            print(label_out)
            print(logits_batch_out)

        print("Accuracy: ")
        print(accuracy_accu / 29)

        coord.request_stop()
        coord.join(threads)
        sess.close()
'''
train_fn()
#flower_eval()
