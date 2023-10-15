"""
A script that trains the explicit network setup in NET_simple - has size 784_30_10 and trained for 30 epochs

The weights and biases are pickle dumped to the files weights and biases
"""

import pickle

from src import mnist_loader
from src.NET_simple import Network

SIZES = [784, 30, 10]
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network(SIZES)

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


def record_weights(net):
    weights = net.weights
    biases = net.biases

    wf = open("weights", 'wb')
    pickle.dump(weights, wf)
    wf.close

    bf = open("biases", 'wb')
    pickle.dump(biases, bf)
    bf.close()

    sm = open("./resources/trainedNets", 'rb')
    try:
        trainedNets = pickle.load(sm)
    except:
        trainedNets = []
    sm.close()
    if ['NONE', "Simple NN"] not in trainedNets:
        sm = open("./resources/trainedNets", 'wb')
        trainedNets.append(['NONE', "Simple NN"])
        pickle.dump(trainedNets, sm)
        sm.close()


record_weights(net)
