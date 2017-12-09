import argparse, sys, os, time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

args = None

def meta_file_at_path(path):
    files = os.listdir(path)
    for file_ in files:
        if file_.endswith('.meta'):
            meta_file = os.path.join(path,file_)
    assert meta_file, 'No metafile found in {}'.format(path)
    return meta_file


def main(_):
    model_variable_scope = args.model
    model_dir = args.log_dir
    meta_file = meta_file_at_path(model_dir)
    saver = tf.train.import_meta_graph(meta_file)
    graph = tf.get_default_graph()
    x= graph.get_tensor_by_name('inputs:0')
    #x = graph.get_tensor_by_name(os.path.join(model_variable_scope,'inputs:0'))
    y_ = graph.get_tensor_by_name('labels:0')
    #y_ = graph.get_tensor_by_name(os.path.join(model_variable_scope,'labels:0'))
    accuracy = graph.get_tensor_by_name('accuracy:0')
    #accuracy = graph.get_tensor_by_name(os.path.join(model_variable_scope,'accuracy:0'))

    x_test = mnist.test.images
    y_labels = mnist.test.labels
    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint(model_dir))
        test_accuracy = sess.run(accuracy,feed_dict={x: x_test, y_: y_labels})
        print('Test set accuracy {}'.format(test_accuracy))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v:  v in ['yes', 'true', 't', 'y', '1'])
    parser.add_argument("--model",type=str,help="variable scope/model name used in running the model")
    parser.add_argument("--log_dir",type=str,default='logs',help="Name of directory to save logs")

    current_path = os.path.abspath('.')
    args, unparsed = parser.parse_known_args()
    mnist = input_data.read_data_sets(os.path.join(current_path,'MNIST_data'), one_hot=True)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
