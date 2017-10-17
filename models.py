import sys
import os

base = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base,'DataSets'))
sys.path.append(os.path.join(base,'Modules'))

# Quick and dirty model training method
# https://sorenbouma.github.io/blog/oneshot/
import numpy as np
import numpy.random as rng
from keras.layers import Input, Conv2D, Lambda, merge, subtract, Dense, Flatten, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from data_gen import generate_siamese_validation_data, siamese_training_data_generator

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

# x_train, y_train = generate_training_data(nums=880)
# x_test, y_test = generate_validation_data()
input_shape = (1, 1084)
left_input = Input(input_shape)
right_input = Input(input_shape)
#build convnet to use in each siamese 'leg'
model = Sequential()
model.add(Dense(64, input_shape=input_shape, activation='relu'))#, kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))#,kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid'))#, kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
model.add(Flatten())
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

nums = 5120
steps = (nums**2)//batch_size
siamese_net.fit_generator(siamese_training_data_generator(nums = nums, input_shape=input_shape),
                        steps, epochs=5,
                        validation_data=generate_siamese_validation_data(input_shape=input_shape)
                        )

print('Final Score : {}'.format(model.evaluate(x_test, y_test, batch_size=16)))
