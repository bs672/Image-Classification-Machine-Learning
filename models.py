import sys
import os

base = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base,'DataSets'))
sys.path.append(os.path.join(base,'Modules'))

# Quick and dirty model training method
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from csv_read import generate_training_data,generate_validation_data

x_train, y_train = generate_training_data()
x_test, y_test = generate_validation_data()

model = Sequential()
model.add(Dense(64, input_shape=(2,1084), activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=10,
          batch_size=128)

print('Final Score : {}'.format(model.evaluate(x_test, y_test, batch_size=16)))
