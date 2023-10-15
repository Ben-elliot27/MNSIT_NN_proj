"""
Script to train and save a model trained on the MNIST data set
"""

import pickle

import tensorflow as tf


def make_model(modelDir, name, model, epochs):
    """
    Train a tf model on the MNIST data set and save it to a file, updating the list of saved models
    :param modelDir: file directory model will be saved to NOTE: this should be in NN_models for file management
    :param name: name of the model
    :param model_layers: tf.keras model of the
    :param epochs: Number of epochs to train for
    :return:
    """

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy'])

    # Check if model name or file name already in use
    used = check_name_file(name, modelDir)
    if used:
        print("Process ended")
        return

    # Get MnistData
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Train model
    model.fit(x_train, y_train, epochs=epochs)

    # Save model
    model.save(modelDir)  # Save the tensorflow model to a file

    # update list of trained models
    update_trained_list(modelDir, name)


def check_name_file(name, modelDir):
    fl = open("./resources/trainedNets", "rb")
    net_data = pickle.load(fl)
    fl.close()
    for net in net_data:
        if net[0] == modelDir or net[1] == name:
            print("File directory or name already in use, would you like to overwrite (y/n)?")
            inp = input("> ")
            if inp == 'y':
                # Overwrite file directory (Should do this automatically when doing code)
                return False
            else:
                # End process
                return True
    else:
        return False


def update_trained_list(modelDir, name):
    # Update list of models
    fl = open("./resources/trainedNets", 'rb')
    try:
        trainedNets = pickle.load(fl)
    except:
        trainedNets = []
    fl.close()
    if [modelDir, name] not in trainedNets:
        sm = open("./resources/trainedNets", 'wb')
        trainedNets.append([modelDir, name])
        pickle.dump(trainedNets, sm)
        sm.close()
    else:
        print("Model already in directory")
