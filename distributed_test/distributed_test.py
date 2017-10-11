# Adapted from Google's introduction to distributed tensorflow pulled 10/07/2017
# URL https://www.tensorflow.org/deploy/distributed

import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



args = None

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

def cluster_spec_dict(should_run_local):
    if should_run_local:
        return {'local':['localhost:2222','localhost:2223']}
    else:
    #host parameter server on VM-3-1 and works on VM-3-2 to VM-3-5
        return {'ps':['10.254.0.36:2222'],'worker':['10.254.0.32:2221', '10.254.0.33:2223','10.254.0.34:2224','10.254.0.35:2225']}

def main(_):

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec(cluster_spec_dict(args.run_local))

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                           job_name=args.job_name,
                           task_index=args.task_index)

    if args.job_name == "ps" or (args.job_name == "local" and args.task_index == 0):
        server.join()
    elif args.job_name == "worker" or (args.job_name == "local" and args.task_index == 1):
    # Assigns ops to the local worker by default.
        x = tf.placeholder(tf.float32, shape=[None, 784])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        with tf.device(tf.train.replica_device_setter(cluster=cluster)):
            #place all variables on parameter server to share
            #create all trainable variables for the model
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            b_fc1 = bias_variable([1024])
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])
            global_step = tf.train.get_or_create_global_step()


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

            cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
            global_step = tf.train.get_or_create_global_step()
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,global_step=global_step)

            cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
            adam_opt = tf.train.AdamOptimizer(1e-4)
        #add op to collect replicas and sync shared parameters

            if args.job_name == 'local':
                is_chief = True
            elif args.task_index == 0:
                is_chief = True
            else:
                is_chief = False
            train_step = adam_opt.minimize(cross_entropy, global_step)
            
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_summ = tf.summary.scalar('train_accuracy',accuracy)
#        summary_op = tf.summary.merge_all()
#        summary_hook = tf.train.SummarySaverHook(save_steps=100, output_dir='logs_distributed_test', summary_writer=None, summary_op=summary_op)
        checkpoint_dir = 'logs_distributed_test'
        trainable_vars = tf.trainable_variables()
#        save_hook=tf.train.CheckpointSaverHook(checkpoint_dir,save_steps=500)
        hooks=[tf.train.StopAtStepHook(num_steps=args.num_steps)]
        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,is_chief=is_chief,checkpoint_dir=checkpoint_dir,hooks=hooks) as sess:
            counter = 0
            while not sess.should_stop():
                batch = mnist.train.next_batch(50)

                if is_chief and counter % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0},session=sess)
                    print('step %d, training accuracy %g' % (counter, train_accuracy))

                sess.run([train_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                counter += 1
            print('Training complete\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v:  v in ['yes', 'true', 't', 'y', '1'])
    parser.add_argument("--num_steps",type=int,default=20000,help="Number of training steps to run.")

    parser.add_argument("--job_name",type=str,default="",help="One of 'ps', 'worker'")
    # args for defining the tf.train.Server
    parser.add_argument("--task_index",type=int,default=0,help="Index of task within the job")
    parser.add_argument("--run_local",type=bool,default=False,help="Pass one of yes, true, t, y, or 1 to run on a single machine.")
    args, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets('/home/ubuntu/project/cs744_project_d3/MNIST_data', one_hot=True)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
