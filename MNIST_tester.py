"""
Script to test the simple explicit NN on the MNIST dataset
"""

import pickle

import numpy as np
from PIL import Image

import mnist_loader
from NET_simple import Network

"""
Script to test the neural network on MNIST images
"""

SIZES = [784, 30, 10]

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network(SIZES)


def read_weights():
    """Function to obtain the weights of the most recent run"""
    wf = open('weights', 'rb')
    weights = pickle.load(wf)
    wf.close()
    bf = open('biases', 'rb')
    biases = pickle.load(bf)
    bf.close()

    return weights, biases


def data_into_net(img_data):
    net = Network(SIZES)
    weights, biases = read_weights()

    net.weights = weights
    net.biases = biases

    test_results = net.feedforward(img_data)  ## NOT WORKING

    print(np.argmax(test_results))


for i, (x, y) in enumerate(test_data):
    if i == 4:
        break
    # Creates PIL image
    mat_img = np.abs((np.array(x) - 1)).reshape(28, 28) * 255
    img = Image.fromarray(np.uint8(mat_img), 'L')
    img.show()

    # Tests image on neural net
    data_into_net(x)
