"""
---------
checking
Python script to test a neural network and recognise a digit

Provides a UI to test Neural networks using a sample image uploaded from the users computer.

Use the netMaker script with make_model() function to create, train and save a model for use in the image test.

Also loads NET_simple a simple 3 layer sequential network

Uses a 28x28 grayscale 255 image - can convert images from other formats and sizes

---------
"""
import os

import pickle
import tkinter as tk
from tkinter import *
from tkinter import filedialog

import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

from NET_simple import Network

SIZES = [784, 30, 10]


def getTrainedNets():
    """
    Func to get all trained nets
    :return:
    trainedNets: a 2D list [['dir', 'name], ...]]
    """
    NN = open("trainedNets", 'rb')
    try:
        trainedNets = pickle.load(NN)
    except:
        trainedNets = []
    NN.close()
    return trainedNets


TRAINED_NNS = getTrainedNets()
nets_names = [el[1] for el in TRAINED_NNS]
nets_dir = [el[0] for el in TRAINED_NNS]

# TKINTER STUFF --------------------------------------------------------------------------------------------------------

my_w = tk.Tk()
my_w.geometry("500x500")  # Size of the window
my_w.title('IMAGE RECOGNITION')
my_font1 = ('times', 18, 'bold')

# Label above button
l1 = tk.Label(my_w, text='Choose type of neural network', font=my_font1)
l1.pack()

# Dropdown to choose neural network
# datatype of menu text
clicked = StringVar()

# initial menu text
clicked.set("None")

# Create Dropdown menu
try:
    drop = OptionMenu(my_w, clicked, *nets_names)  # List of neural networks
    drop.pack()
except TypeError:
    l1.config(text="No Trained NNs")

b1_2 = tk.Button(my_w, text='Get Summary of NN architecture',
                 command=lambda: setSummary(l1_2, clicked.get()))
b1_2.pack()

l1_2 = tk.Label(my_w, text="Summary of NN", font=('times', 15, 'bold'))  # width?
l1_2.pack()

l_a = tk.Label(my_w, text="""------------------------------------------------------------------------------------------
                          """, font=my_font1)
l_a.pack()

l2 = tk.Label(my_w, text='Upload Files for ML recognition', font=my_font1, )
l2.pack()

# Button to upload file
b1 = tk.Button(my_w, text='Upload Files', command=lambda: controller(clicked.get(), l3))
b1.pack()

l3 = tk.Label(my_w, text="Prediction from NN", font=my_font1)
l3.pack()


# TKINTER STUFF END -----------------------------------------------------------------------------------------------------

def setSummary(l1_2, clicked):
    # Check if a valid NN is selected
    if clicked == "None":
        l1_2.config(text="Please choose a valid NN", fg="#FF0000")
        return
    else:
        try:
            model = getNets(nets_dir[nets_names.index(clicked)])
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            net_info_txt = "\n".join(stringlist)
        except AttributeError:
            net_info_txt = "SIMPLE NN: 784 - 30 - 10, 30 epochs"
        l1_2.config(text=net_info_txt, fg="#000000")


def controller(clicked, l3):
    """
    clicked: type of neural network chosen
    l1: label 1 in tkinter menu
    Main controller loop of the program
    :return:
    """
    # Check valid NN
    if clicked == "None":
        return
    img_data = upload_file()

    # Run data through correct NN
    NET = getNets(nets_dir[nets_names.index(clicked)])

    if NET != False:
        # RUN TF NN
        img_data = convData(np.array(img_data))
        output = NET.predict(img_data)
        probabilities = tf.nn.softmax(output).numpy()
        result = np.argmax(probabilities, axis=1)
    else:
        # RUN simple NN
        result = data_into_net_simple(img_data)
    l3.config(text=str(result))


def upload_file():
    """
    Upload a file from the users system and returns the list of greyscale pictures from the 28x28 scaled image
    :return: data: the image data in the form of MNIST dataset
    """
    f_types = [('Jpg Files', '*.jpg'),
               ('PNG Files', '*.png'), ('Jpeg Files', '*.jpeg')]  # type of files to select
    filename = tk.filedialog.askopenfilename(multiple=True, filetypes=f_types)
    col = 1  # start from column 1
    row = 3  # start from row 3
    for f in filename:
        img = Image.open(f).convert('L')  # read the image file
        img = img.resize((28, 28))  # new width & height
        # img=img.resize((30,784)) # new width & height

        WIDTH, HEIGHT = img.size
        data = list(img.getdata())  # convert image data to a list of integers

        img = ImageTk.PhotoImage(img)
        e1 = tk.Label(my_w)
        e1.pack()
        e1.image = img  # keep a reference! by attaching it to a widget attribute
        e1['image'] = img  # Show Image

        data = np.array(data) / 255  # Normalise data
        data = np.abs(data - 1)  # put it with black 1, white 0
        data = data.reshape(len(data), 1)  # make every element itself a list
        data = data.tolist()

        return data


def data_into_net_simple(img_data):
    net = Network(SIZES)
    weights, biases = read_weights()

    net.weights = weights
    net.biases = biases

    test_results = net.feedforward(img_data)  ## NOT WORKING

    return np.argmax(test_results)


def read_weights():
    """Function to obtain the weights of the most recent run"""
    wf = open('weights', 'rb')
    weights = pickle.load(wf)
    wf.close()
    bf = open('biases', 'rb')
    biases = pickle.load(bf)
    bf.close()

    return weights, biases


def getNets(dir):
    """
    A function to get the different NNs stored and return their info
    dir: directory to file containing tensorflow model
    -- 'NONE' if it is the non tf simple model
    :return:
    if dir != 'NONE':
    returns a tensorflow model
    else returns false
    """
    if dir == 'NONE':
        return False
    else:
        return tf.keras.models.load_model(dir)


def convData(img_data):
    # Function to get image data in tkinter neural network form
    img_data_list = []
    img_data_list.append(np.array(img_data).reshape(28, 28).tolist())

    return img_data_list


my_w.mainloop()  # Keep the window open
