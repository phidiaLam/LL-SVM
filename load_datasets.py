import os
import numpy as np
import struct
import pandas as pd
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
            magic, n = struct.unpack('>II', lbpath.read(8))
            train_labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(train_images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            train_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(train_labels), 784)
        with open(test_labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            test_labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(test_images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            test_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(test_labels), 784)

        return train_images, train_labels, test_images, test_labels

    def load_usps(self):
        pass

    def load_letter(self):
        print("start load mnist dataset")
        train_labels_path = os.path.join('./datasets/mnist/', 'train-labels.idx1-ubyte')
        train_images_path = os.path.join('./datasets/mnist/', 'train-images.idx3-ubyte')
        test_labels_path = os.path.join('./datasets/mnist/', 't10k-labels.idx1-ubyte')
        test_images_path = os.path.join('./datasets/mnist/', 't10k-images.idx3-ubyte')
        with open(train_labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            train_labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(train_images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            train_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(train_labels), 784)
        with open(test_labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            test_labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(test_images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            test_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(test_labels), 784)


        return train_images, train_labels, test_images, test_labels

    def load_caltech(self):
        pass

