import tensorflow as tf # tensorflow module
import numpy as np # numpy module
import os # path join


DATA_DIR = "../dataset/"
TRAIN_FILE = "train.tfrecord"
VALIDATION_FILE = "validation.tfrecord"
TRAINING_SET_SIZE = 3200
N_EPOCHS = 5
BATCH_SIZE = 10
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


def decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features=FEATURES_LIST)
    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=N_CHANNEL)
    #image_object = _image_object()
    image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    #image_object.image = image
    #image_object.height = features["image/height"]
    #image_object.width = features["image/width"]
    #image_object.filename = features["image/filename"]
    label = tf.cast(features["image/class/label"], tf.int64)
    #image_object.label = label
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
    #dataset = dataset.map(augment)
    #dataset = dataset.map(normalize)

    #the parameter is the queue size
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

def inference_fn(image_batch):
    #x = tf.placeholder(tf.float32, [None, flat_len])
    x_image = tf.reshape(image_batch, [-1, 1536*1536*3])
    W = tf.Variable(tf.zeros([1536*1536*3, 4]))
    b = tf.Variable(tf.zeros([4]))

    y = tf.nn.softmax(tf.matmul(x_image, W) + b)

    return y


def train_fn():
    with tf.Graph().as_default():
        image_batch, label_batch = inputs(train=True, batch_size = BATCH_SIZE
                                                        , num_epochs=N_EPOCHS)

        image_batch_placeholder = tf.placeholder(tf.float32
                                , shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNEL])
        image_batch = tf.reshape(image_batch
                                , (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, N_CHANNEL))

        label_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_CLASSES])
        label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
        label_batch_one_hot = tf.one_hot(tf.add(label_batch, label_offset)
                                         , depth=N_CLASSES, on_value=1.0, off_value=0.0)

        logits = inference_fn(image_batch_placeholder)
    #    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_one_hot, logits=logits))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_batch_placeholder, logits=logits))

        train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

        #saver = tf.train.Saver()

        with tf.Session() as sess:
            # Visualize the graph through tensorboard.
            file_writer = tf.summary.FileWriter("./logs", sess.graph)

            sess.run(tf.global_variables_initializer())
            #saver.restore(sess, "dltmp/checkpoint-train.ckpt")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            try:
                step = 0
                while True:
                    image_out, label_out, label_batch_one_hot_out = sess.run([image_batch, label_batch, label_batch_one_hot])

                    _, infer_out, loss_out = sess.run([train_step, logits, loss], feed_dict={image_batch_placeholder: image_out, label_batch_placeholder: label_batch_one_hot_out})

                    if step%10 == 0:
                        print(step)
                        #print(image_out.shape)
                        #print("label_out: ")
                        #print(filename_out)
                        #print(label_out)
                        #print(label_batch_one_hot_out)
                        #print("infer_out: ")
                        #print(infer_out)
                        print("loss: ", loss_out)
                    #if(i%50 == 0):
                    #    saver.save(sess, "dltmp/checkpoint-train.ckpt")
                    step +=1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' % (N_EPOCHS, step))

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

if __name__ == '__main__':
    train_fn()
    #flower_eval()
