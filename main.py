import sys

from algorithm_compute import train_test
from load_datasets import LoadDatasets
import numpy as np



if __name__ == '__main__':
    load = LoadDatasets()
    dataset = sys.argv[1]
    anchor_number = int(sys.argv[2])
    if type(anchor_number)!=int or anchor_number<=0:
        print("Anchor Number need integer")
        exit(1)
    print("start load datasets")
    X_train, y_train, X_test, y_test = None, None, None, None
    match dataset:
        case 'mnist':
            X_train, y_train, X_test, y_test = load.load_mnist()
        case 'usps':
            X_train, y_train, X_test, y_test = load.load_usps()
        case 'letter':
            X_train, y_train, X_test, y_test = load.load_letter()
        case 'caltech':
            X_train, y_train, X_test, y_test = load.load_caltech()
        case _:
            print("Can not found dataset")
            exit(1)
    train_test(X_train, y_train, X_test, y_test, anchor_number)

