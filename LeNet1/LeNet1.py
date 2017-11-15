#Implementation fo LeNet1
#from LeCun Y, Boser BE, Denker JS, Henderson D, Howard RE, Hubbard WE, Jackel LD. Handwritten digit recognition with a back-propagation network. InAdvances in neural information processing systems 1990 (pp. 396-404).

import argparse, sys, os, time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

args = None

"""Define the cluster used for distributed traing.  You can pass true to run training locally"""
def cluster_spec_dict(should_run_local):
    if should_run_local:
        return {'local':['localhost:2222','localhost:2223']}
    else:
    #host parameter server on VM-3-1 and works on VM-3-2 to VM-3-5
        return {'ps':['10.254.0.36:2222'],'worker':['10.254.0.32:2221', '10.254.0.33:2223','10.254.0.34:2224','10.254.0.35:2225']}

################### Model specific funtions ############################

def scaled_tanh(x):
    #f(a) = Atanh(Sa)
    return tf.multiply(1.7159,tf.tanh(tf.multiply(2/3,x)))

def model(x):
    with tf.variable_scope('lenet1'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        h1 = tf.layers.conv2d(x_image,filters=4,kernel_size=[5,5],padding='VALID',activation=tf.nn.relu,name='h1')
        h2 = tf.layers.average_pooling2d(h1, pool_size=[2,2],strides=[2,2],padding='VALID',name='h2')
        h3 = tf.layers.conv2d(h2,filters=12,kernel_size=[5,5],padding='VALID',activation=tf.nn.relu,name='h3')
        h4 = tf.layers.average_pooling2d(h3, pool_size=[2,2],strides=[2,2],padding='VALID',name='h4')
        h4_flat = tf.reshape(h4,[-1,192])
        y_logits = tf.layers.dense(h4_flat,10,activation=tf.nn.sigmoid,name='outputs')

        #add all neurons to collection
        tf.add_to_collection('neurons',h1)
        tf.add_to_collection('neurons',h2)
        tf.add_to_collection('neurons',h3)
        tf.add_to_collection('neurons',h4)
    return y_logits

def make_model(cluster):
    with tf.device(tf.train.replica_device_setter(cluster=cluster)):
        x = tf.placeholder(tf.float32, shape=[None, 784],name='inputs')
        y_ = tf.placeholder(tf.float32, shape=[None, 10],name='labels')
        y_logits = model(x)
        global_step = tf.train.get_or_create_global_step()

        return x, y_,y_logits,  global_step
###############################################################################


def main(_):

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec(cluster_spec_dict(args.run_local))

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,job_name=args.job_name,task_index=args.task_index)

    if args.job_name == "ps" or (args.job_name == "local" and args.task_index == 0):
        server.join()
    elif args.job_name == "worker" or (args.job_name == "local" and args.task_index == 1):

        #load module for passed model script
        x, y_,y_logits,global_step = make_model(cluster)
        cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_logits),name='loss')
        loss_summ = tf.summary.scalar('loss',cross_entropy)
        adam_opt = tf.train.AdamOptimizer(1e-3)
        train_step = adam_opt.minimize(cross_entropy, global_step)

        #add a summary measuring accuracy
        correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
        accuracy_summ = tf.summary.scalar('train_accuracy',accuracy)
        summary_ops = tf.summary.merge_all()
        checkpoint_dir = args.log_dir
        init_op = tf.global_variables_initializer()
        if args.job_name == 'local':
            is_chief = True
        elif args.task_index == 0:
            is_chief = True
        else:
            is_chief = False


        if is_chief:
            print("Worker {}: Initializing session...".format(args.task_index))
            saver = tf.train.Saver(name='lenet1_saver')
        else:
            print("Worker {}: Waiting for session to be initialized...".format(args.task_index))
        sv = tf.train.Supervisor(is_chief=is_chief,init_op=init_op,summary_op=None,saver=None,recovery_wait_secs=1,global_step=global_step)
        sess = sv.prepare_or_wait_for_session(server.target)
        print("Worker {}: Session initialization complete.".format(args.task_index))

        if is_chief:
            summary_writer = tf.summary.FileWriter(checkpoint_dir,sess.graph)

        start_time = time.time()
        while True:
            batch = mnist.train.next_batch(100)
            _, _global_step = sess.run([train_step, global_step],feed_dict={x: batch[0], y_: batch[1]})
            if _global_step % 400 == 0:
                train_accuracy, _summary_ops = sess.run([accuracy, summary_ops],feed_dict={x: batch[0], y_: batch[1]})
                print('step {}, training accuracy {}'.format(_global_step, train_accuracy))
                if is_chief:
                    saver.save(sess, os.path.join(checkpoint_dir,'model.ckpt'))
                    summary_writer.add_summary(_summary_ops,_global_step)

            if _global_step > args.num_steps:
                break
        duration = time.time() - start_time
        print('Training complete\nRun time {} seconds.'.format(duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v:  v in ['yes', 'true', 't', 'y', '1'])
    parser.add_argument("--num_steps",type=int,default=20000,help="Number of training steps to run.")
    parser.add_argument("--run_local",type=bool,default=False,help="Pass one of yes, true, t, y, or 1 to run on a single machine.")
    parser.add_argument("--log_dir",type=str,default='logs',help="Name of directory to save logs")

    # args for defining the tf.train.Server
    parser.add_argument("--job_name",type=str,default="",help="One of 'ps', 'worker'")
    parser.add_argument("--task_index",type=int,default=0,help="Index of task within the job")
    current_path = os.path.abspath('.')
    args, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets(os.path.join(current_path,'MNIST_data'), one_hot=True)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
