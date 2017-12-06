#Reimplementation of DeepXplore
#Sources:
#Pei K, Cao Y, Yang J, Jana S. DeepXplore: Automated Whitebox Testing of Deep Learning Systems. arXiv preprint arXiv:1705.06640. 2017 May 18.
#

import argparse, sys, os, time, json, random
import importlib
from collections import OrderedDict

import numpy as np
from scipy.misc import imsave
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        try:
            os.mkdir(directory)
        except FileNotFoundError as err:
            print('Could not create directory: {}.\nError was: {}\nCheck directory structure.'.format(directory,err))


def create_models(config,x):

    """Loads pretrained models to the default object graph for use with DeepXplore.  Also performs the following.
    1.  Grab handles to the model input, output, and intermediate output layer tensors
    3.  Create and initialize a coverage dictionary for each model"""
    graph = tf.get_default_graph()
    models = {}
    model_locations = config['models']
    for model_name, model_info in model_locations.items():
        import_name = model_info['import_name']
        #import the module
        module = importlib.import_module(import_name)
        y = module.model(x)
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_name)
        var_dict = {_var.name.split(':')[0]:_var for _var in vars_list}
        saver = tf.train.Saver(var_list=var_dict)
        neurons = graph.get_collection('neurons',scope=model_name)
        coverage_dict = create_coverage_dict(neurons)
        #class_prob = tf.reduce_max(y,axis =1)
        models[model_name] = {'saver':saver,'outputs':y,'neurons':neurons,'coverage_dict':coverage_dict}

    return models


def rescale_raw_data(x):
    x_max = np.max(x)
    return x/x_max

def raw_data_to_image(x):
    scaled = rescale_raw_data(x)
    scaled *= 255
    scaled = scaled.astype('uint8')
    return scaled.reshape(28,28)

def neuron_coverage(coverage_dict):
    covered_neurons = sum([bool_val for bool_val in coverage_dict.values() if bool_val])
    total_neurons = len(coverage_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

def choose_random_neuron(coverage_dict):
    not_covered = [(layer_tensor,index) for (layer_tensor,index), bool_val in coverage_dict.items() if not bool_val]
    if not_covered:
        layer_tensor, tensor_index = random.choice(not_covered)
    else:
        layer_tensor, tensor_index = random.choice(coverage_dict.keys())
    return layer_tensor, tensor_index

def neuron_fired(model_dict,layer, tensor_index,x,input_data, threshold):
    layer_output = layer.eval(feed_dict={x:input_data})
    max_layer_out = np.max(layer_output,axis=0)
    return max_layer_out[tensor_index] > threshold

def neuron(layer,tensor_index,input_data):
    batch_size = input_data.shape[0]
    tensor_list = list(tensor_index)
    tensor_list.insert(0,batch_size - 1)
    neuron = layer[tensor_list]
    return neuron

def create_coverage_dict(neurons):
    coverage_dict = OrderedDict()
    for layer in neurons:
        starting_index = layer.shape
        create_layer_coverage(layer,coverage_dict)
    return coverage_dict

def create_layer_coverage(layer,coverage_dict,tensor_indices=None, tensor_level=0):
    shape = layer.shape.as_list()[1:]
    if not tensor_indices:
        tensor_indices = shape
    for i in range(0,shape[tensor_level]):
        tensor_indices[tensor_level] = i
        if tensor_level == len(shape)-1:
            coverage_dict[(layer,tuple(tensor_indices))] = False
        else:
            create_layer_coverage(layer,coverage_dict,tensor_indices=tensor_indices,tensor_level=tensor_level+1)

def update_coverage(x,gen_img, models, threshold):
    for model_name, model_dict in models.items():
        neurons = model_dict['neurons']
        coverage_dict = model_dict['coverage_dict']
        for layer in neurons:
            layer_output = layer.eval(feed_dict={x:gen_img})
            #take max accross dimension 0
            max_layer_out = np.max(layer_output,axis =0)
            above_threshold_mask = np.greater(max_layer_out,threshold)
            it = np.nditer(above_threshold_mask,flags=['multi_index'])
            while not it.finished:
                tensor_index = it.multi_index
                coverage_dict[(layer,tensor_index)]=it[0]
                it.iternext()

def constrain_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads


def model_predictions(models,x, gen_img):
    predictions = []
    pred_model_labels = []
    for model_name,model_dict in models.items():
        outputs = model_dict['outputs']
        prediction = np.argmax(outputs.eval(feed_dict={x:gen_img}),axis=1)
        predictions.append(prediction[0])
        pred_model_labels.append(model_name)
    return predictions, pred_model_labels

def main(_):
    x = tf.placeholder(tf.float32, shape=[None, 784],name='inputs')
    models = create_models(config,x)
    target_model = config['target_model']
    threshold = config['threshold']
    transformation = config['transformation']
    step_size = config['step_size']
    lambda1 = tf.constant(config['lambda1'])
    lambda2 = tf.constant(config['lambda2'])
    clipping = config['clipping']
    seeds = config['seeds']

    #define all computed losses and gradients before initialization of the graph
    non_target_models = set(models.keys())
    non_target_models.difference_update(set([target_model]))
    target_model_loss = tf.multiply(tf.negative(models[target_model]['outputs']),lambda1)
    non_target_losses = tf.add_n([models[model]['outputs'] for model in non_target_models])
    obj1_loss = tf.add(non_target_losses,target_model_loss)

    for seed in range(seeds):
        #gen_img = mnist.test.next_batch(args.batch_size)[0]
        gen_img = mnist.test.next_batch(1)[0]
        orig_img = gen_img.copy()

        neurons_to_cover = []
        for model_name,model_dict in models.items():
            coverage_dict = model_dict['coverage_dict']
            layer, tensor_index = choose_random_neuron(coverage_dict)
            neurons_to_cover.append(neuron(layer,tensor_index,gen_img))
        neuron_sum = tf.add_n(neurons_to_cover)
        obj2_loss = tf.reduce_mean(tf.add(obj1_loss,tf.multiply(neuron_sum,lambda2)))
        grads = tf.gradients(obj2_loss,x)[0]
        with tf.Session() as sess:
            #load trained weights for all graphs
            for model_name,model_dict in models.items():
                saver = model_dict['saver']
                model_dir = config['models'][model_name]['model_dir']
                saver.restore(sess,tf.train.latest_checkpoint(model_dir))


            predictions, pred_model_labels = model_predictions(models,x, gen_img)
            if not len(set(predictions)) == 1:
                #classification is already different in the networks.
                #in original DeepXplore these images are added to the difference inducing list.  We will skip and focus on images created by transformation
                continue
            #shared_label = np.zeros((1,10))
            #shared_label[(0,predictions[0])] = 1
            shared_label = predictions[0]
            #choose a random neuron from each network currently not covered


            #run gradient ascent for 20 steps
            for step in range(0,20):
                gradients = grads.eval(feed_dict={x:gen_img})
                if transformation == 'light':
                    gradients = constrain_light(gradients)
                if transformation == 'contrast':
                    gradients = constrain_contrast(gradients)
                gen_img += gradients*step_size
                if clipping == True:
                    gen_img = np.clip(gen_img,0,1)
                predictions, pred_model_labels = model_predictions(models,x,gen_img)
                if not len(set(predictions)) == 1:
                    update_coverage(x,gen_img, models, threshold)
                    final_gen_img = raw_data_to_image(gen_img)
                    final_orig_img = raw_data_to_image(orig_img)
                    new_label = list(set(predictions).difference(set([shared_label])))[0]
                    final_gen_img_name = '{}_seed_{}_label_{}_to_{}.png'.format(target_model,seed,shared_label,new_label)
                    final_orig_img_name = '{}_seed_{}_orig.png'.format(target_model,seed)
                    imsave(os.path.join(out_dir,final_gen_img_name),final_gen_img)
                    imsave(os.path.join(out_dir,final_orig_img_name),final_orig_img)
                    break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v:  v in ['yes', 'true', 't', 'y', '1'])
    parser.add_argument("--config_json",type=str,help="json file to use for loading trained models and specifying model parameters.")
    parser.add_argument("--out_dir",type=str,default='xplore_out',help="Name of directory to save DeepXplore model and outputs")
    parser.add_argument("--transformation",type=str,help="Type of transformation to apply. One of 'light', 'TO DO'")
    parser.add_argument("--seeds",type=int,default=2000,help="Integer number of seed examples.")
    args, unparsed = parser.parse_known_args()
    out_dir = args.out_dir
    ensure_dir_exists(out_dir)
    assert args.config_json, 'You must specify a .json file specifying the models to use for DeepXplore.'
    seeds = args.seeds
    with open(args.config_json,'r') as fh:
        config = json.load(fh)

    current_path = os.path.abspath('.')
    mnist = input_data.read_data_sets(os.path.join(current_path,'MNIST_data'), one_hot=True)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
