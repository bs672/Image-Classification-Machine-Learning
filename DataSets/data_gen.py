import numpy as np
from csv_read import load_extracted_features, load_graph, load_seed, load_seed_matrix, load_extracted_features_PCA, load_CCA_features

class DataGen():
    # Training is true if for training, else it is for validation/evaluation/prediction
    def __init__(self, data, labels, batch_size, input_shape, nums=None, training=True, siamese=True):
        self.data = data
        self.labels = labels
        self.nums
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.training = training
        self.siamese = siamese

    def generator():
        pass


def siamese_2D_data_generator(nums=5120, batch_size=128, input_shape =(32,32,1)):
    g = load_graph(shape_match = True)[:nums,:nums]
    f = load_extracted_features_PCA(k=input_shape[0]*input_shape[1])[:nums] # set the first 5120 features for training
    f.shape = (nums, input_shape[0], input_shape[1], 1)
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

def siamese_2D_validation_data(nums=880, input_shape =(32,32,1)):
    g = load_graph(shape_match = True)[(6000-nums):,(6000-nums):]
    f = load_extracted_features_PCA(k=input_shape[0]*input_shape[1])[(6000-nums):6000] # set the first 5120 features for training
    f.shape = (nums, input_shape[0], input_shape[1], 1)
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

# Makes a data generator for classification
def siamese_training_data_generator(nums=5120,batch_size=128,input_shape=(1,1084)):
    g = load_graph(shape_match = True)[:nums,:nums]
    f = load_extracted_features_PCA(k=input_shape[0]*input_shape[1])[:nums] # set the first 5120 features for training
    input1 = np.empty(
        (batch_size, ) + input_shape)
    input2 = np.empty(
        (batch_size, ) + input_shape)
    outputs = np.empty(
        (batch_size, ) )
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

def generate_siamese_validation_data(nums=880, input_shape=(1,1084)):
    g = load_graph(shape_match = True)[(6000-nums):,(6000-nums):]
    f = load_extracted_features_PCA(k=input_shape[0]*input_shape[1])[(6000-nums):6000] # set the first 5120 features for training
    input1 = np.empty(
        (nums**2,  ) + input_shape )
    input2 = np.empty(
        (nums**2,  ) + input_shape )
    outputs = np.empty(
        (nums**2, ) )
    batch_index = 0
    for i in range(nums):
        for j in range(nums):
            input1[batch_index], input2[batch_index] = f[i] , f[j]
            outputs[batch_index] = g[i, j]
            batch_index += 1
    return [input1, input2] ,outputs

def seed_generator(batch_size=128,input_shape=(1,1084)):
    g = load_seed_matrix()
    f = load_CCA_features(k = input_shape[0]*input_shape[1], onlyseeds=True)
    if f is None:
        print('You must compute CCA for {} features'.format(f))
    nums = f.shape[0]
    input1 = np.empty(
        (batch_size, ) + input_shape)
    input2 = np.empty(
        (batch_size, ) + input_shape)
    outputs = np.empty(
        (batch_size, ) )
    while True:
        i_array = np.random.choice(nums, batch_size)
        j_array = np.random.choice(nums, batch_size)
        for batch_index in range(batch_size):
            i, j = i_array[batch_index], j_array[batch_index]
            input1[batch_index], input2[batch_index] = f[i] , f[j]
            outputs[batch_index] = g[i, j]
        yield [input1, input2] ,outputs

def seed_validation(input_shape=(1,1084)):
    g = load_seed_matrix()
    f = load_CCA_features(k = input_shape[0]*input_shape[1], onlyseeds=True)
    if f is None:
        print('You must compute CCA for {} features'.format(f))
    nums = f.shape[0]
    input1 = np.empty(
        (nums**2,  ) + input_shape )
    input2 = np.empty(
        (nums**2,  ) + input_shape )
    outputs = np.empty(
        (nums**2, ) )
    batch_index = 0
    for i in range(nums):
        for j in range(nums):
            input1[batch_index], input2[batch_index] = f[i] , f[j]
            outputs[batch_index] = g[i, j]
            batch_index += 1
    return [input1, input2] ,outputs

# runs a sanity check on a siamese need, ie a = a ==> 1
def siamese_sanity_check(input_shape=(1,1084)):
    f = load_CCA_features(k = input_shape[0]*input_shape[1])
    nums = f.shape[0]
    input1 = np.empty(
        (nums,  ) + input_shape )
    input2 = np.empty(
        (nums,  ) + input_shape )
    outputs = np.empty(
        (nums, ) )
    for i in range(nums):
        input1[i], input2[i] = f[i] , f[i]
        outputs[i] = 1
    return [input1, input2] ,outputs
