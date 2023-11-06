import os
import numpy as np
import struct
import pandas as pd
import h5py
from functools import reduce
'''
Load the data sets from file
return: X_train, y_train, X_test, y_test
'''
class LoadDatasets:
    def load_mnist(self):
        print("start load mnist dataset")
        train_labels_path = os.path.join('./datasets/mnist/', 'train-labels.idx1-ubyte')
        train_images_path = os.path.join('./datasets/mnist/', 'train-images.idx3-ubyte')
        test_labels_path = os.path.join('./datasets/mnist/', 't10k-labels.idx1-ubyte')
        test_images_path = os.path.join('./datasets/mnist/', 't10k-images.idx3-ubyte')
        with open(train_labels_path, 'rb') as lbpath:
            train_labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(train_images_path, 'rb') as imgpath:
            train_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(train_labels), 784)
        with open(test_labels_path, 'rb') as lbpath:
            test_labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(test_images_path, 'rb') as imgpath:
            test_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(test_labels), 784)

        return train_images, train_labels, test_images, test_labels

    def load_usps(self):
        with h5py.File(os.path.dirname(os.path.abspath(__file__))+"./datasets/usps/usps.h5", 'r') as hf:
            train = hf.get('train')
            train_images = train.get('data')[:]
            train_labels = train.get('target')[:]
            test = hf.get('test')
            test_images = test.get('data')[:]
            test_labels = test.get('target')[:]

            train_images = train_images.reshape(train_images.shape[0], reduce(lambda a, b: a * b, train_images.shape[1:]))
            test_images = test_images.reshape(test_images.shape[0], reduce(lambda a, b: a * b, test_images.shape[1:]))
        return train_images, train_labels, test_images, test_labels


    def load_letter(self):
        print("start load mnist dataset")
        train_labels_path = os.path.join('./datasets/letter/', 'emnist-letters-train-labels-idx3-ubyte')
        train_images_path = os.path.join('./datasets/letter/', 'emnist-letters-train-images-idx3-ubyte')
        test_labels_path = os.path.join('./datasets/letter/', 'emnist-letters-test-labels-idx3-ubyte')
        test_images_path = os.path.join('./datasets/letter/', 'emnist-letters-test-images-idx3-ubyte')
        with open(train_labels_path, 'rb') as lbpath:
            train_labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(train_images_path, 'rb') as imgpath:
            train_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(train_labels), 784)
        with open(test_labels_path, 'rb') as lbpath:
            test_labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(test_images_path, 'rb') as imgpath:
            test_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(test_labels), 784)


        return train_images, train_labels, test_images, test_labels

    def load_caltech(self):
        pass

