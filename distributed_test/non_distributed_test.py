# adapted from tensorflow tutorial on convolutional nets with mnist accessed 10/7/2017
# URL https://www.tensorflow.org/get_started/mnist/pros
import sys, os, argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def get_meta_file(logs_dir):
    log_files = os.listdir(logs_dir)
    latest_step = 0
    metafile = None
    for file_ in log_files:
        if '.meta' in file_:
            step = int(file_.split('-')[1].split('.')[0])
            if step > latest_step:
                metafile = file_

    return metafile

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def build_model():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    #create all trainable variables for the model
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    #create layers using weights from above
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return x, keep_prob, y_, y_conv

def run_training(x, keep_prob, y_, y_conv):
        cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
        global_step = tf.train.get_or_create_global_step()
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summ = tf.summary.scalar('train_accuracy',accuracy)
        tf.summary.merge_all()
        logs_dir = 'logs_non_distributed_test'
        hooks=[tf.train.StopAtStepHook(num_steps=args.num_steps)]
        with tf.train.MonitoredTrainingSession(hooks=hooks, checkpoint_dir=logs_dir,save_summaries_steps=100) as sess:
            counter = 0
            while not sess.should_stop():
                batch = mnist.train.next_batch(50)

                if counter % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0},session=sess)
                    print('step %d, training accuracy %g' % (counter, train_accuracy))

                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5},session=sess)
                counter += 1
            print('Training complete\n')
        with tf.Session() as sess:
            metafile = get_meta_file(logs_dir)
            new_saver = tf.train.import_meta_graph(os.path.join(logs_dir,metafile))
            new_saver.restore(sess, tf.train.latest_checkpoint(logs_dir))
            _accuracy_test = accuracy.eval(session=sess,feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})
            print('test_set accuracy: {:02.4f}'.format(_accuracy_test,4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # args for defining the tf.train.ClusterSpec
    parser.add_argument("--num_steps",type=int,default=20000,help="Number of training steps to run.")
    args, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    x, keep_prob, y_, y_conv = build_model()
    run_training(x, keep_prob, y_, y_conv)
