import sys, os, argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import resnet_model as rn

MODEL_DIR = './model_dir/resnet'
X_FEATURE = 'x'
NUM_CLASSES = 10
INIT_LEARNING_RATE = 0.5

def build_mnist_resnet(inputs, resnet_size=38, is_training=True):
    '''
    Build resnet model for mnist.

    params
    ------
    is_training: boolean
        needed for batch_normalization

    returns
    -------
    output layer of resnet
    '''

    if resnet_size % 6 != 2:
        raise ValueError('resnet_size must be 6n + 2:', resnet_size)

    num_blocks = (resnet_size - 2) // 6
    
    # data format required for tensorflow's resnet model
    data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    # construct 4 blocks of different width and height dimensions
    inputs = rn.conv2d_fixed_padding(
        inputs=inputs, filters=16, kernel_size=3, strides=1,
        data_format=data_format)
    inputs = tf.identity(inputs, 'initial_conv')

    inputs = rn.block_layer(
        inputs=inputs, filters=16, block_fn=rn.building_block, blocks=num_blocks,
        strides=1, is_training=is_training, name='block_layer1',
        data_format=data_format)
    inputs = rn.block_layer(
        inputs=inputs, filters=32, block_fn=rn.building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer2',
        data_format=data_format)
    inputs = rn.block_layer(
        inputs=inputs, filters=64, block_fn=rn.building_block, blocks=num_blocks,
        strides=2, is_training=is_training, name='block_layer3',
        data_format=data_format)

    inputs = rn.batch_norm_relu(inputs, is_training, data_format)
    inputs = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=7, strides=1, padding='VALID',
        data_format=data_format)
    inputs = tf.identity(inputs, 'final_avg_pool')
    inputs = tf.reshape(inputs, [-1, 64])
    inputs = tf.layers.dense(inputs=inputs, units=NUM_CLASSES)
    inputs = tf.identity(inputs, 'final_dense')

    # each block has multiple units
    # each unit has a 1-channel layer, multi-channel layer and 1-channel layer
    return inputs

def resnet_model_fn(features, labels, mode, params):
    '''
    params
    ------
    params: Currently not utilized. Options are hardcoded
    
    returns
    -------
    spec: EstimatorSpec object
        contains mode, prediction, loss, train_op, eval_metric_ops
    '''
    x = features[X_FEATURE]
    # reshape x
    if len(x.shape) < 4:
        x = tf.reshape(x, [-1, 28, 28, 1])

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    
    # build model
    logits = build_mnist_resnet(x, resnet_size=38, is_training=is_training)

    # generate predictions
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)
    
    # define loss
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)

    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    
    loss = cross_entropy # add other terms for regularization
    
    # define training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=INIT_LEARNING_RATE)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # batch norm requires this
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else: # evaluate mode
        train_op = None
    
    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

def main():
    # read data
    mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={X_FEATURE: mnist.train.images},
        y=mnist.train.labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={X_FEATURE: mnist.test.images},
        y=mnist.test.labels,
        num_epochs=1,
        shuffle=False)

    # create Estimator
    mnist_resnet_classifier = tf.estimator.Estimator(
        model_fn=resnet_model_fn, model_dir=MODEL_DIR)

    tensors_to_log = {
        'train_accuracy':'train_accuracy',
        'cross_entropy':'cross_entropy'
    }

    # Run
    for _ in range(10):
        mnist_resnet_classifier.train(
            input_fn=train_input_fn,
            steps=100)

        eval_results = mnist_resnet_classifier.evaluate(
            input_fn=test_input_fn)

        print(eval_results)

if __name__ == '__main__':
    main()
