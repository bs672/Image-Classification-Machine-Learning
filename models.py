import sys
import os

base = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base,'DataSets'))
sys.path.append(os.path.join(base,'Modules'))

# Quick and dirty model training method
# https://sorenbouma.github.io/blog/oneshot/
import numpy as np
import numpy.random as rng
from keras.layers import Input, Conv2D, Lambda, subtract, Dense, Flatten, Dropout, GaussianNoise
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from data_gen import seed_generator, seed_validation, siamese_sanity_check, seed_similarity_generator, siamese_sanity_check_symmetry
from csv_read import load_seed, load_CCA_features

FNAME = 'test'
# Test2 has greater gaussian noise and more dropout
# x_train, y_train = generate_training_data(nums=880)
# x_test, y_test = generate_validation_data()
k = 8
input_shape = (1, k)
left_input = Input(input_shape)
right_input = Input(input_shape)

# Internal model is a softmax 10-class classifier
#TEST
model = Sequential()
model.add(GaussianNoise(0.001, input_shape=input_shape))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#encode each of the two inputs into a vector with the convnet
encoded_l, encoded_r = model(left_input), model(right_input)
#merge two encoded inputs with the l1 distance between them
L1_distance = lambda x: K.abs(x[0]-x[1])
# Merge is depreciated, will need subtract
subtracted = subtract([encoded_l,encoded_r])
both = Lambda(lambda x: K.abs(x))(subtracted)
# both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1,activation='sigmoid')(both)
siamese_net = Model(inputs=[left_input,right_input], outputs=prediction)

batch_size = 128

siamese_net.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

# model.fit(x_train, y_train,
#           epochs=10,
#           batch_size=128)
override = False
if os.path.isfile(FNAME+'.hdf5') and not override:
    siamese_net.load_weights(FNAME+'.hdf5')
else:
    steps = 10000
    siamese_net.fit_generator(seed_generator(batch_size=batch_size, input_shape=input_shape),
                            steps, epochs=10,
                            validation_data=seed_validation(input_shape=input_shape))

x_test, y_test = siamese_sanity_check(input_shape=input_shape)
print('Performing sanity check: ([x,x]) => 1')
sanity_check = siamese_net.evaluate(x_test, y_test, batch_size=10)
print('\nSanity Check : {}'.format('Passed' if sanity_check[1] == 1.0 else 'Failed'))

x_test, y_test = siamese_sanity_check_symmetry(input_shape=input_shape)
print('Performing sanity check: ([y,x]) => ([x,y])')
sanity_check = siamese_net.evaluate(x_test, y_test, batch_size=10)
print('\nSanity Check : {}'.format('Passed' if sanity_check[1] == 1.0 else 'Failed'))

if not os.path.isfile(FNAME+'.hdf5') or override:
    siamese_net.save_weights(FNAME+'.hdf5')

seeds = load_seed()
seed_features = load_CCA_features(k = 8, onlyseeds=True)
features = load_CCA_features(k = 8)
nums = features.shape[0]
s = seed_features.shape[0]
input1 = np.empty(
    (s, ) + input_shape)
input2 = np.empty(
    (s, ) + input_shape)
labels_max = np.zeros((nums,2), dtype = 'int')
labels_sum = np.zeros((nums,2), dtype = 'int')
confidence = np.zeros((nums,2), dtype = 'float')
for batch in range(nums):
    # initialize input_1
    for i in range(s):
        input1[i], input2[i] = seed_features[i], features[batch]
    pred = siamese_net.predict([input1, input2], batch_size=60)
    # print('Predictions for data point {}: {}'.format(batch+1, pred))
    similarity_sums = np.zeros(10)
    for j in range(s):
        similarity_sums[seeds[j][1]] += pred[j][0]
    # if index == (batch + 1):
    #     print("Seed {} with label {} prediction values".format(index, label))
    #     print('')
    # print(similarity_sums)
    labels_sum[batch] = np.array([batch+1, np.argmax(similarity_sums)], dtype='int')
    labels_max[batch] = np.array([batch+1, seeds[np.argmax(pred)][1]], dtype='int')
    # confidence[batch] = batch+1, np.max(pred)
    confidence[batch] = batch+1, np.max(similarity_sums)/6.0
print('Performing sanity check: Perfect Seed Labeling')

correctness_max = 0.0
correctness_sum = 0.0
for index, label in seeds:
    assert index == labels_max[index-1][0] == labels_sum[index-1][0]
    # print('{} should be {}'.format(labels[index-1][1],label))
    correctness_max += int(labels_max[index-1][1] == label)
    correctness_sum += int(labels_sum[index-1][1] == label)

correctness_max /= s
print('Percentage of correctly clustered seeds via Max Arg: {}'.format(correctness_max))
correctness_sum /= s
print('Percentage of correctly clustered seeds via Sum: {}'.format(correctness_sum))

print('Confidence in each prediction via max similarity')
for index, c in confidence:
    print('Element {} has confidence {:1.3f}'.format(int(index),c))
np.savetxt('AllLabels{}viaSum.csv'.format(FNAME), np.asarray(labels_sum), delimiter=',', fmt='%d')
np.savetxt('results{}viaSum.csv'.format(FNAME), np.asarray(labels_sum[6000:]), delimiter=',', fmt='%d', header='Id,Label')
np.savetxt('AllLabels{}viaMax.csv'.format(FNAME), np.asarray(labels_max), delimiter=',', fmt='%d')
np.savetxt('results{}viaMax.csv'.format(FNAME), np.asarray(labels_max[6000:]), delimiter=',', fmt='%d', header='Id,Label')
