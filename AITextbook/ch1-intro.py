import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#Define layers using Sequential - specify what each layer looks like inside that call
#Dense means neurons are fully connected - all neurons connected to all neurons in next layer
neurons = Dense(units=1, input_shape=[1])
model = Sequential([neurons])
model.compile(optimizer='sgd', loss='mean_squared_error')

#Training model to find relationship between these two arrays
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)
#Relationship is 2x-1 so it should give something very close to 19 (it does)
print(model.predict([10.0]))
#This line will have it print what it found to be the weight (x) and bias (b) in the y=mx+b equation
print("Here's what I learned: {}".format(neurons.get_weights()))