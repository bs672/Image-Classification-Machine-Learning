import numpy as np
from csv_read import load_extracted_features, load_graph, load_seed

# Files for generating data
# Makes a data generator for classification
def generate_training_data(nums=5120):
    g = load_graph(shape_match = True)[:nums,:nums]
    f = load_extracted_features()[:nums] # set the first 5120 features for training
    input_shape = (2,1084)
    inputs = np.empty(
        (nums**2, ) + input_shape)
    outputs = np.empty(
        (nums**2, ))
    batch_index = 0
    for i in range(nums):
        for j in range(nums):
            inputs[batch_index] = np.array([f[i] , f[j]])
            outputs[batch_index] = g[i, j]
            batch_index += 1
    return inputs,outputs

def generate_siamese_training_data(nums=5120, input_shape = (1084)):
    g = load_graph(shape_match = True)[:nums,:nums]
    f = load_extracted_features()[:nums] # set the first 5120 features for training
    input1 = np.empty(
        (nums**2, ) + input_shape)
    input2 = np.empty(
        (nums**2, ) + input_shape)
    outputs = np.empty(
        (nums**2, ))
    batch_index = 0
    for i in range(nums):
        for j in range(nums):
            input1[batch_index], input2[batch_index] = f[i] , f[j]
            outputs[batch_index] = g[i, j]
            batch_index += 1
    return input1, input2 ,outputs

# Makes a data generator for classification
def siamese_training_data_generator(nums=5120,batch_size=128,input_shape =(1,1084)):
    g = load_graph(shape_match = True)[:nums,:nums]
    f = load_extracted_features()[:nums] # set the first 5120 features for training
    input1 = np.empty(
        (batch_size, ) + input_shape)
    input2 = np.empty(
        (batch_size, ) + input_shape)
    outputs = np.empty(
        (batch_size, ))
    batch_index, i, j = 0, 0, 0
    while True:
        input1[batch_index], input2[batch_index] = f[i] , f[j]
        outputs[batch_index] = g[i, j]
        batch_index += 1
        i += 1
        if not (i % nums):
            i = 0
            j = (j + 1) % nums
        if batch_index >= batch_size:
            batch_index = 0
            yield [input1, input2] ,outputs

def training_data_generator(nums=5120,batch_size=128):
    g = load_graph(shape_match = True)[:nums,:nums]
    f = load_extracted_features()[:nums] # set the first 5120 features for training
    input_shape = (2,1084)
    inputs = np.empty(
        (batch_size, ) + input_shape)
    outputs = np.empty(
        (batch_size, ))
    batch_index, i, j = 0, 0, 0
    while True:
        inputs[batch_index] = np.array([f[i] , f[j]])
        outputs[batch_index] = g[i, j]
        batch_index += 1
        i += 1
        if not (i % nums):
            i = 0
            j = (j + 1) % nums
        if batch_index >= batch_size:
            batch_index = 0
            yield inputs, outputs

def generate_validation_data(nums=880):
    g = load_graph(shape_match = True)[(6000-nums):,(6000-nums):]
    f = load_extracted_features()[(6000-nums):6000] # set the first 5120 features for training
    input_shape = (2,1084)
    inputs = np.empty(
        (nums**2, ) + input_shape)
    outputs = np.empty(
        (nums**2, ))
    batch_index = 0
    for i in range(nums):
        for j in range(nums):
            inputs[batch_index] = np.array([f[i] , f[j]])
            outputs[batch_index] = g[i, j]
            batch_index += 1

    return inputs,outputs

def generate_siamese_validation_data(nums=880, input_shape = (1,1084)):
    g = load_graph(shape_match = True)[(6000-nums):,(6000-nums):]
    f = load_extracted_features()[(6000-nums):6000] # set the first 5120 features for training
    input1 = np.empty(
        (nums**2, ) + input_shape)
    input2 = np.empty(
        (nums**2, ) + input_shape)
    outputs = np.empty(
        (nums**2, ))
    batch_index = 0
    for i in range(nums):
        for j in range(nums):
            input1[batch_index], input2[batch_index] = f[i] , f[j]
            outputs[batch_index] = g[i, j]
            batch_index += 1
    return [input1, input2] ,outputs
