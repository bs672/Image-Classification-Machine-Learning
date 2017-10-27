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
from data_gen import seed_generator, seed_validation, siamese_sanity_check

# x_train, y_train = generate_training_data(nums=880)
# x_test, y_test = generate_validation_data()
k = 8
input_shape = (1, k)
left_input = Input(input_shape)
right_input = Input(input_shape)

# Internal model is a softmax 10-class classifier
model = Sequential()
# model.add(Dense(64, input_shape=input_shape, activation='relu'))
# model.add(Flatten())
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
if os.path.isfile('test.hdf5') and not override:
    siamese_net.load_weights('test.hdf5')
else:
    steps = 10000
    siamese_net.fit_generator(seed_generator(batch_size=batch_size, input_shape=input_shape),
                            steps, epochs=5,
                            validation_data=seed_validation(input_shape=input_shape))

x_test, y_test = siamese_sanity_check(input_shape=input_shape)
sanity_check = siamese_net.evaluate(x_test, y_test, batch_size=10)
print('Final Score : {}'.format(sanity_check))

if sanity_check[1] < 1.0:
    print('Siamese Net isn\'t working')
elif not os.path.isfile('test.hdf5') or override:
    siamese_net.save_weights('test.hdf5')
