"""
    Implementation of Neural Network Modules,
    Including the data generator functions for training, evaluation, and prediction
"""

import sys
import os
import numpy as np
from itertools import combinations
from math import factorial
from keras.layers import Input, Conv2D, Lambda, subtract, Dense, Flatten, Dropout, GaussianNoise
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

# extracts the feature for seeds
def features_of_seeds(s, features):
    output = np.zeros((s.shape[0],features.shape[1]))
    for i in range(s.shape[0]):
        output[i] = features[s[i][0]-1]
    return output

# returns the seed matrix of seeds s
def seed_matrix(s):
    nums = s.shape[0]
    output = np.zeros((nums, nums), dtype='int')
    for i in range(nums):
        for j in range(nums):
            label1, label2 = s[i,1], s[j,1]
            output[i,j] = int(s[i,1]==s[j,1])
    assert check_symmetric(output), 'Seed Matrix should be symmetric'
    return output

class SiameseModel():

    def __init__(self, data, seeds, name, dense=False, batch_size = 256):
        # input dimensions (size of input, eg 10)
        # data: all featurs
        # seed: index and labels of seed
        # name, from which the model is saved to
        self.input_shape = (1, data.shape[1])
        self.X = data # features of seeds
        self.X_seeds = features_of_seeds(seeds, data) # gets the seed features
        self.seeds = seeds
        self.s_matrix = seed_matrix(seeds)
        self.fname = os.path.join('Modules', (name + '.hdf5'))
        self.label_name = name + '.csv  '
        self.batch_size = batch_size
        self.dense = dense # Do we use the combinations for the model?


        self.model = None

    def create_model(self, from_save = True):
        left_input = Input(self.input_shape)
        right_input = Input(self.input_shape)

        # Internal model is a softmax 10-class classifier
        internalmodel = Sequential()
        internalmodel.add(GaussianNoise(0.0001, input_shape=self.input_shape))
        internalmodel.add(Flatten())
        internalmodel.add(Dense(64, activation='relu'))
        internalmodel.add(Dropout(0.5))
        internalmodel.add(Dense(64, activation='relu'))
        internalmodel.add(Dropout(0.5))
        internalmodel.add(Dense(10, activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # TODO, compare to ADAM
        internalmodel.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        encoded_l, encoded_r = internalmodel(left_input), internalmodel(right_input)
        subtracted = subtract([encoded_l,encoded_r])
        both = Lambda(lambda x: K.abs(x))(subtracted)
        prediction = Dense(1,activation='sigmoid')(both)
        siamese_net = Model(inputs=[left_input,right_input], outputs=prediction)
        # TODO, compare to ADAM
        siamese_net.compile(loss='binary_crossentropy',
                            optimizer='rmsprop',
                            metrics=['accuracy'])

        if from_save and os.path.isfile(self.fname):
            siamese_net.load_weights(self.fname)

        self.model = siamese_net

    # fits the model
    def fit(self, steps = 10000, epochs = 10, save=True):
        if self.model is None:
            print('Must create model first')
            self.create_model()

        self.model.fit_generator(self.training_generator(), steps, epochs=epochs,
                                validation_data=self.validation_generator())

        if save:
            self.model.save_weights(self.fname)

    # Runs sanity checks on the model
    def evaluate(self):
        print('Performing sanity check: ([x,x]) => 1')
        print('\nSanity Check : {}'.format('Passed' if self.siamese_sanity_check() == 1.0 else 'Failed'))

        print('Performing sanity check: ([y,x]) => ([x,y])')
        print('Sanity Check : {}'.format('Passed' if self.symmetry_sanity_check() == 1.0 else 'Failed'))

    # Runs predictions on similarity to seeds for the data
    def predict(self, split=False, via_max=True, verbose=True, save=True):
        if self.model is None:
            print('Must create model first')
            self.create_model()
            print('Checking invariants')
            self.evaluate()
        # produces all comparsions of one feature to all seeds
        input1 = np.empty((self.X_seeds.shape[0], ) + self.input_shape)
        input2 = np.empty((self.X_seeds.shape[0], ) + self.input_shape)
        labels = np.zeros((self.X.shape[0],2), dtype='int')
        # loop through the number of elements to check similarity
        for batch in range(self.X.shape[0]):
            # initialize input_1
            for i in range(self.X_seeds.shape[0]):
                input1[i], input2[i] = self.X_seeds[i], self.X[batch]
            pred = self.model.predict([input1, input2], batch_size=self.X_seeds.shape[0])
            if via_max:
                labels[batch] = np.array([batch+1, self.seeds[np.argmax(pred)][1]], dtype='int')
            else: #via sum
                similarity_sums = np.zeros(10)
                for j in range(self.X_seeds.shape[0]):
                    similarity_sums[self.seeds[j][1]] += pred[j][0]
                labels[batch] = np.array([batch+1, np.argmax(similarity_sums)], dtype='int')
        if verbose: print('Performing sanity check: Perfect Seed Labeling')

        correctness = 0.0
        for index, label in self.seeds:
            assert index == labels[index-1][0]
            correctness += int(labels[index-1][1] == label)

        correctness /= self.X_seeds.shape[0]
        if verbose: print('Percentage of correctly clustered seeds via {}: {}'.format('Max' if via_max else 'Sum', correctness))

        if split and 10000 == self.X.shape[0]:
            if verbose: print('Returning all labels and the last 4000')
            if save:
                print('Also saving them')
                np.savetxt('Allvia{}-{}'.format('Max' if via_max else 'Sum',self.label_name), np.asarray(labels), delimiter=',', fmt='%d')
                np.savetxt('Resultsvia{}-{}'.format('Max' if via_max else 'Sum',self.label_name), np.asarray(labels[6000:]), delimiter=',', fmt='%d')
            return labels, labels[6000:]
        else:
            if verbose: print('Returning {} labels'.format(labels.shape[0]))
            if save:
                print('Also saving them')
                np.savetxt('via{}-{}'.format('Max' if via_max else 'Sum',self.label_name), np.asarray(labels), delimiter=',', fmt='%d')
            return labels

    def training_generator(self):
        nums = self.X_seeds.shape[0]
        input1 = np.empty(
            (self.batch_size, ) + self.input_shape)
        input2 = np.empty(
            (self.batch_size, ) + self.input_shape)
        outputs = np.empty(
            (self.batch_size, ) )
        while True:
            i_array = np.random.choice(nums, self.batch_size)
            j_array = np.random.choice(nums, self.batch_size)
            for batch_index in range(self.batch_size):
                i, j = i_array[batch_index], j_array[batch_index]
                input1[batch_index], input2[batch_index] = self.X_seeds[i] , self.X_seeds[j]
                outputs[batch_index] = self.s_matrix[i, j]
            yield [input1, input2] ,outputs

    # generates all seeds
    def validation_generator(self):
        nums = self.X_seeds.shape[0]
        input1 = np.empty(
            (nums**2,  ) + self.input_shape )
        input2 = np.empty(
            (nums**2,  ) + self.input_shape )
        outputs = np.empty(
            (nums**2, ) )
        batch_index = 0
        for i in range(nums):
            for j in range(nums):
                input1[batch_index], input2[batch_index] = self.X_seeds[i] , self.X_seeds[j]
                outputs[batch_index] = self.s_matrix[i, j]
                batch_index += 1
        return [input1, input2] ,outputs

    # produces pairs of the same element, which should all be similar
    def siamese_sanity_check(self):
        nums = self.X.shape[0]
        input1 = np.empty(
            (nums,  ) + self.input_shape )
        input2 = np.empty(
            (nums,  ) + self.input_shape )
        outputs = np.empty(
            (nums, ) )
        for i in range(nums):
            input1[i], input2[i] = self.X[i] , self.X[i]
            outputs[i] = 1
        return self.model.evaluate([input1, input2], outputs, batch_size=10)[1]

    # runs predictions are pairs of two, and then the reverse pair, trying to get
    # the same result
    # same result given the [y,x] == [x,y] property
    def symmetry_sanity_check(self, steps=10000):
        passed = 0
        input1 = np.empty(
            (steps,  ) + self.input_shape )
        input2 = np.empty(
            (steps,  ) + self.input_shape )
        i_array = np.random.choice(self.X_seeds.shape[0], steps)
        j_array = np.random.choice(self.X_seeds.shape[0], steps)
        for k in range(steps):
            i, j = i_array[k], j_array[k]
            input1[k], input2[k] = self.X_seeds[i], self.X_seeds[j]
        x = self.model.predict([input1, input2], batch_size = steps)
        y = self.model.predict([input2, input1], batch_size = steps)
        return 1.0 - (np.sum(np.abs(x-y)) / steps)

def to_sparse_labels(labels):
    assert labels.shape[1] == 2
    output = np.zeros((labels.shape[0], 10), dtype='int')
    for i in range(labels.shape[0]):
        output[i,labels[i,1]] = 1
    assert np.max(output) == 1
    return output

def to_dense_labels(labels, index_range = np.arange(6001, 10001)):
    assert labels.shape == (4000,10), str(labels)
    output = np.zeros((labels.shape[0], 2), dtype='int')
    for i in range(labels.shape[0]):
        output[i,0] = index_range[i]
        output[i,1] = np.argmax(labels[i,:])
        # assert np.sum(labels[i,:]) == 1, print(labels[i,:])
    return output

class Classifier():

    def __init__(self, data, labels, name, sep=5500, batch_size = 256):
        # input dimensions (size of input, eg 10)
        # data : all features (10000)
        # sep: separator for training, validation data
        # lables : labels for training and validation data
        self.input_shape = (data.shape[1])
        self.output_shape = (10)
        self.training_data = data[:sep]
        self.validation_data = data[sep:labels.shape[0]]
        self.labels = to_sparse_labels(labels)
        self.evaluation_data = data[:labels.shape[0]]
        self.prediction_data = data[labels.shape[0]:]
        self.fname = os.path.join('Modules', (name + '.hdf5'))
        self.sep = sep
        self.label_name = name + '.csv'
        self.batch_size = batch_size

        assert data.shape[0] == 10000
        assert self.prediction_data.shape[0] == 4000
        assert self.labels.shape == (6000,10)
        assert labels.shape == (6000,2)
        self.model = None

    def create_model(self, from_save = True):

        # Uses internal model of siamese Network
        # model is a softmax 10-class classifier
        internalmodel = Sequential()
        internalmodel.add(Dense(64, activation='relu', input_dim=self.input_shape))
        internalmodel.add(Dropout(0.25))
        internalmodel.add(Dense(64, activation='relu'))
        internalmodel.add(Dropout(0.25))
        internalmodel.add(Dense(10, activation='softmax'))
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        # TODO, compare to ADAM
        internalmodel.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        if from_save and os.path.isfile(self.fname):
            internalmodel.load_weights(self.fname)

        self.model = internalmodel

    # fits the model
    def fit(self, steps = 10000, epochs = 20, save=True):
        if self.model is None:
            print('Must create model first')
            self.create_model()

        print('Training with {} points of data, validating with {}'.format(self.sep, self.validation_data.shape[0]))
        self.model.fit_generator(self.training_generator(), steps, epochs=epochs,
                                validation_data=self.validation_generator(), validation_steps = steps // 10)

        if save:
            self.model.save_weights(self.fname)

    def evaluate(self):
        if self.model is None:
            print('Must create model first')
            self.create_model()
        self.model.evaluate(self.evaluation_data, self.labels, batch_size=6000, verbose=1)
    # Runs predictions on similarity to seeds for the data
    def predict(self, steps = 100, verbose=True, save=True):
        if self.model is None:
            print('Must create model first')
            self.create_model()
        # produces all comparsions of one feature to all seeds
        labels = self.model.predict_generator(self.prediction_generator(steps=steps), steps)
        print(labels.shape)

        labels = to_dense_labels(labels)
        if verbose: print('Returning labels')
        if save:
            print('Also saving them')
            np.savetxt('{}'.format(self.label_name), np.asarray(labels), delimiter=',', fmt='%d',  header='Id,Label')
        return labels

    def training_generator(self):
        inputs = np.empty(
            (self.batch_size, self.input_shape) )
        outputs = np.empty(
            (self.batch_size, self.output_shape) )
        while True:
            i_array = np.random.choice(self.sep, self.batch_size)
            for batch_index in range(self.batch_size):
                inputs[batch_index] = self.training_data[i_array[batch_index]]
                outputs[batch_index] = self.labels[i_array[batch_index]]
            yield inputs ,outputs

    # generates all seeds
    def validation_generator(self):
        inputs = np.empty(
            (self.batch_size, self.input_shape ) )
        outputs = np.empty(
            (self.batch_size, self.output_shape ) )
        while True:
            i_array = np.random.choice(self.validation_data.shape[0], self.batch_size)
            for batch_index in range(self.batch_size):
                inputs[batch_index] = self.validation_data[i_array[batch_index]]
                outputs[batch_index] = self.labels[i_array[batch_index] + self.sep]
            yield inputs ,outputs

    def prediction_generator(self, steps = 100, batch = 40):
        inputs = np.empty(
            (batch, self.input_shape ) )
        assert (steps * batch) == 4000
        while True:
            batch_index = 0
            for i in range(batch):
                inputs[batch_index] = self.prediction_data[batch_index]
                batch_index += 1
            yield inputs


# Tries methods of handling overfitting by using n choose k linear combinations of the seed
class FatSiameseModel():

    def __init__(self, data, seeds, name, dense=False, batch_size = 256):
        # input dimensions (size of input, eg 10)
        # data: all featurs
        # seed: index and labels of seed
        # name, from which the model is saved to
        self.input_shape = (1, data.shape[1])
        # Original Data
        self.X = data # features of seeds
        self.X_seeds = features_of_seeds(seeds, data) # gets the seed features
        self.seeds = seeds
        self.s_matrix = seed_matrix(seeds)
        self.fname = os.path.join('Modules', (name + '.hdf5'))
        self.label_name = name + '.csv'
        self.batch_size = batch_size
        self.dense = dense # Do we use the combinations for the model?

        # Propogated data
        self.fat_X_seeds, self.fat_s_matrix = None, None
        self.propagate()

        # print(self.fat_s_matrix)

        self.model = None

    # expands the data based on the seeds and linear combinations
    def propagate(self):
        # gets a 10 x 6 matrix of each labels seed
        table = np.zeros((10,6),dtype='int')
        counters = [0] * 10
        for i, val in enumerate(self.seeds):
            table[val[1], counters[val[1]]] = i
            counters[val[1]] += 1
        assert all([i == 6 for i in counters]), str(counters)
        assert np.min(table) >= 0 and np.max(table) < 60, str(table)

        # precalced the size, 630 per label * 10 --> 630
        # self.fat_X_seeds = np.zeros((630, self.X_seeds.shape[1]))
        self.fat_s_matrix = np.zeros((630, 630), dtype='int') # is a block diagonal matrix
        output = []
        for label in range(10):
            a, b = 63 * (label), 63 * (1+label)
            assert len(output) == a, '{} {} {}'.format(a, len(output), label)
            for r in range(1,7):
                for index_set in combinations(table[label],r):
                    avg = np.zeros(self.X_seeds.shape[1])
                    for i in index_set:
                        avg += self.X_seeds[i]
                    output.append(avg/r)

            assert len(output) == b, str(output)
            self.fat_s_matrix[a:b, a:b] = 1

        self.fat_X_seeds = np.array(output)
        assert self.fat_X_seeds.shape == (630,self.X_seeds.shape[1]), str(self.fat_X_seeds)
        assert np.sum(self.fat_s_matrix) == (63*63*10)




    def create_model(self, from_save = True):
        left_input = Input(self.input_shape)
        right_input = Input(self.input_shape)

        # Internal model is a softmax 10-class classifier
        internalmodel = Sequential()
        internalmodel.add(GaussianNoise(0.0001, input_shape=self.input_shape))
        internalmodel.add(Flatten())
        internalmodel.add(Dense(64, activation='relu'))
        internalmodel.add(Dropout(0.5))
        internalmodel.add(Dense(64, activation='relu'))
        internalmodel.add(Dropout(0.5))
        internalmodel.add(Dense(10, activation='softmax'))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # TODO, compare to ADAM
        internalmodel.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        encoded_l, encoded_r = internalmodel(left_input), internalmodel(right_input)
        #merge two encoded inputs with the l1 distance between them
        L1_distance = lambda x: K.abs(x[0] - x[1])
        subtracted = subtract([encoded_l,encoded_r])
        both = Lambda(lambda x: K.abs(x))(subtracted)
        prediction = Dense(1,activation='sigmoid')(both)
        siamese_net = Model(inputs=[left_input,right_input], outputs=prediction)
        # TODO, compare to ADAM
        siamese_net.compile(loss='binary_crossentropy',
                            optimizer='rmsprop',
                            metrics=['accuracy'])

        if from_save and os.path.isfile(self.fname):
            siamese_net.load_weights(self.fname)

        self.model = siamese_net

    # fits the model
    def fit(self, steps = 10000, epochs = 10, save=True):
        if self.model is None:
            print('Must create model first')
            self.create_model()

        self.model.fit_generator(self.training_generator(), steps, epochs=epochs,
                                validation_data=self.validation_generator(), validation_steps=steps//10)

        if save:
            self.model.save_weights(self.fname)

    # Runs sanity checks on the model
    def evaluate(self):
        if self.model is None:
            print('Must create model first')
            self.create_model()
        print('Performing sanity check: ([x,x]) => 1')
        print('\nSanity Check : {}'.format('Passed' if self.siamese_sanity_check() == 1.0 else 'Failed'))

        print('Performing sanity check: ([y,x]) => ([x,y])')
        sane = self.symmetry_sanity_check()
        print('Sanity Check : {}'.format('Passed' if sane == 1.0 else 'Failed --> ' + str(sane)))

    # Runs predictions on similarity to seeds for the data
    def predict(self, split=False, via_max=True, verbose=True, save=True):
        if self.model is None:
            print('Must create model first')
            self.create_model()
            print('Checking invariants')
            self.evaluate()
        # produces all comparsions of one feature to all seeds
        input1 = np.empty((self.X_seeds.shape[0], ) + self.input_shape)
        input2 = np.empty((self.X_seeds.shape[0], ) + self.input_shape)
        labels = np.zeros((self.X.shape[0],2), dtype='int')
        # loop through the number of elements to check similarity
        for batch in range(self.X.shape[0]):
            # initialize input_1
            for i in range(self.X_seeds.shape[0]):
                input1[i], input2[i] = self.X_seeds[i], self.X[batch]
            pred = self.model.predict([input1, input2], batch_size=self.X_seeds.shape[0])
            if via_max:
                labels[batch] = np.array([batch+1, self.seeds[np.argmax(pred)][1]], dtype='int')
            else: #via sum
                similarity_sums = np.zeros(10)
                for j in range(s):
                    similarity_sums[self.seeds[j][1]] += pred[j][0]
                labels[batch] = np.array([batch+1, np.argmax(similarity_sums)], dtype='int')
        if verbose: print('Performing sanity check: Perfect Seed Labeling')

        correctness = 0.0
        for index, label in self.seeds:
            assert index == labels[index-1][0]
            correctness += int(labels[index-1][1] == label)

        correctness /= self.X_seeds.shape[0]
        if verbose: print('Percentage of correctly clustered seeds via {}: {}'.format('Max' if via_max else 'Sum', correctness))

        if split and 10000 == self.X.shape[0]:
            if verbose: print('Returning all labels and the last 4000')
            if save:
                print('Also saving them')
                np.savetxt('All{}'.format(self.label_name), np.asarray(labels), delimiter=',', fmt='%d')
                np.savetxt('Results{}'.format(self.label_name), np.asarray(labels[6000:]), delimiter=',', fmt='%d',  header='Id,Label')
            return labels, labels[6000:]
        else:
            if verbose: print('Returning {} labels'.format(labels.shape[0]))
            if save:
                print('Also saving them')
                np.savetxt('{}'.format(self.label_name), np.asarray(labels), delimiter=',', fmt='%d',  header='Id,Label')
            return labels

    # modified to use fat seeds
    def training_generator(self):
        nums = self.fat_X_seeds.shape[0]
        input1 = np.empty(
            (self.batch_size, ) + self.input_shape)
        input2 = np.empty(
            (self.batch_size, ) + self.input_shape)
        outputs = np.empty(
            (self.batch_size, ) )
        while True:
            i_array = np.random.choice(nums, self.batch_size)
            j_array = np.random.choice(nums, self.batch_size)
            for batch_index in range(self.batch_size):
                i, j = i_array[batch_index], j_array[batch_index]
                input1[batch_index], input2[batch_index] = self.fat_X_seeds[i] , self.fat_X_seeds[j]
                outputs[batch_index] = self.fat_s_matrix[i, j]
            yield [input1, input2] ,outputs

    # generates from all seeds, only validates on real seeds
    def validation_generator(self):
        nums = self.X_seeds.shape[0]
        input1 = np.empty(
            (nums**2,  ) + self.input_shape )
        input2 = np.empty(
            (nums**2,  ) + self.input_shape )
        outputs = np.empty(
            (nums**2, ) )
        batch_index = 0
        for i in range(nums):
            for j in range(nums):
                input1[batch_index], input2[batch_index] = self.X_seeds[i] , self.X_seeds[j]
                outputs[batch_index] = self.s_matrix[i, j]
                batch_index += 1
        return [input1, input2] ,outputs

    # produces pairs of the same element, which should all be similar
    def siamese_sanity_check(self):
        nums = self.X.shape[0]
        input1 = np.empty(
            (nums,  ) + self.input_shape )
        input2 = np.empty(
            (nums,  ) + self.input_shape )
        outputs = np.empty(
            (nums, ) )
        for i in range(nums):
            input1[i], input2[i] = self.X[i] , self.X[i]
            outputs[i] = 1
        return self.model.evaluate([input1, input2], outputs, batch_size=10)[1]

    # runs predictions are pairs of two, and then the reverse pair, trying to get
    # the same result
    # same result given the [y,x] == [x,y] property
    # TODO, change so all predictions are done at once
    def symmetry_sanity_check(self, steps=10000):
        passed = 0
        input1 = np.empty(
            (steps,  ) + self.input_shape )
        input2 = np.empty(
            (steps,  ) + self.input_shape )
        i_array = np.random.choice(self.fat_X_seeds.shape[0], steps)
        j_array = np.random.choice(self.fat_X_seeds.shape[0], steps)
        for k in range(steps):
            i, j = i_array[k], j_array[k]
            input1[k], input2[k] = self.fat_X_seeds[i], self.fat_X_seeds[j]
        x = self.model.predict([input1, input2], batch_size = steps)
        y = self.model.predict([input2, input1], batch_size = steps)
        return 1.0 - (np.sum(np.abs(x-y)) / steps)
