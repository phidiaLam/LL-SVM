from core_algorithm import init, train_test
from load_datasets import LoadDatasets
import pandas as pd
import numpy as np


def replace_number(labels, n):
    array = []
    for index, value in enumerate(labels):
        if value == n:
            array.append(1.0)
        else:
            array.append(-1.0)
    return array

# init()
if __name__ == '__main__':
    load = LoadDatasets()
    X_train, y_train, X_test, y_test = load.load_usps()
    y_train = np.where(y_train == 1, 1, -1)
    y_test = np.where(y_test == 1, 1, -1)

    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)

    train_test(X_train, y_train, X_test, y_test)