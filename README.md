# MNSIT_NN_proj
## Number recognition from NN trained on MNIST

This project contains scripts to train tf neural networks of any shape on MNIST data and saves them to a file 
so that it can be used in the image tester.

The image tester loads saved neural netowrks (NN) and provides the user a UI to choose a saved NN, get the shape
summary of it and upload an image from thier own machine which they can test the neural network on.

The project also contains an explicit simple 3 layer neural network (NET_simple) which can be loaded and trained on MNIST using the NET_simple_loader script.

Uses code and builds on work from:
http://neuralnetworksanddeeplearning.com/chap1.html
Neural networks and deep learning  Nielsen, Michael A


To train a network use the netMaker script and function def make_model(modelDir, name, model, epochs).
To load the UI for image testing run the image_test script.

External Libraries: Tensorflow, Pickle, Numpy

