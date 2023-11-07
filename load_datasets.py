import os
import numpy as np
import struct
import pandas as pd
import h5py
from functools import reduce
from PIL import Image
import cv2

from sklearn.model_selection import train_test_split
'''
Load the data sets from file
return: X_train, y_train, X_test, y_test
'''
class LoadDatasets:
    def load_mnist(self):
        print("start load mnist dataset")
        train_labels_path = os.path.dirname(os.path.abspath(__file__)) + os.path.join('/datasets/mnist/', 'train-labels.idx1-ubyte')
        train_images_path = os.path.dirname(os.path.abspath(__file__)) + os.path.join('/datasets/mnist/', 'train-images.idx3-ubyte')
        test_labels_path = os.path.dirname(os.path.abspath(__file__)) + os.path.join('/datasets/mnist/', 't10k-labels.idx1-ubyte')
        test_images_path = os.path.dirname(os.path.abspath(__file__)) + os.path.join('/datasets/mnist/', 't10k-images.idx3-ubyte')
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
        print("start load usps dataset")
        with h5py.File(os.path.dirname(os.path.abspath(__file__))+"/datasets/usps/usps.h5", 'r') as hf:
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
        print("start load letter dataset")
        train_labels_path = os.path.dirname(os.path.abspath(__file__)) + os.path.join('/datasets/letter/', 'emnist-letters-train-labels-idx1-ubyte')
        train_images_path = os.path.dirname(os.path.abspath(__file__)) + os.path.join('/datasets/letter/', 'emnist-letters-train-images-idx3-ubyte')
        test_labels_path = os.path.dirname(os.path.abspath(__file__)) + os.path.join('/datasets/letter/', 'emnist-letters-test-labels-idx1-ubyte')
        test_images_path = os.path.dirname(os.path.abspath(__file__)) + os.path.join('/datasets/letter/', 'emnist-letters-test-images-idx3-ubyte')
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
        data = []
        dataset_directory = os.path.dirname(os.path.abspath(__file__)) + '/datasets/caltech'

        for class_folder in os.listdir(dataset_directory):
            class_folder_path = os.path.join(dataset_directory, class_folder)
            if os.path.isdir(class_folder_path):
                for image_file in os.listdir(class_folder_path):
                    if image_file.lower().endswith(('.jpg', '.png')):
                        image_path = os.path.join(class_folder_path, image_file)
                        class_label = class_folder
                        image = load_and_preprocess_image(image_path)
                        data.append((image, class_label))

        # Split data into features and labels
        X = np.array([item[0] for item in data])
        y = np.array([item[1] for item in data])

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, y_train, X_test,  y_test


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image, dtype=np.float32)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure correct color channel order
    image = image / 255.0  # Normalize to the range [0, 1]
    return image
