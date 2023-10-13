

"""
checking
A sample script for creating a tf model, train it on MNIST data and save it to the list of trained nets
"""

import pickle
import numpy as np
import tensorflow as tf

import mnist_loader

modelDir = "NN_models/SeqNet_784_128_drop02_10__5"

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

x_train_1 = []
y_train_1 = []

# Getting data in correct form
for (x, y) in training_data:
    x_train_1.append(x.reshape(28, 28).tolist())
    y_train_1.append(y.reshape(10).tolist())

for i, arr in enumerate(y_train_1):
    y_train_1[i] = np.argmax(arr)

x_train_1 = np.array(x_train_1)
y_train_1 = np.array(y_train_1)

x_test_1 = []
for (x, y) in test_data:
    x_test_1.append(x.reshape(28, 28).tolist())

x_test_1 = np.array(x_test_1)

# Set up neural network

model_1 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model_1.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy'])

model_1.fit(x_train_1, y_train_1, epochs=5)

model_1.save(modelDir)  # Save the tensorflow model to a file

# Update list of models
sm = open("trainedNets", 'rb')
try:
    trainedNets = pickle.load(sm)
except:
    trainedNets = []
sm.close()
if [modelDir, "Sequential TF Net"] not in trainedNets:
    sm = open("trainedNets", 'wb')
    trainedNets.append([modelDir, "Sequential TF Net"])
    pickle.dump(trainedNets, sm)
    sm.close()
