from core_algorithm import init, train_test
from load_datasets import LoadDatasets
import pandas as pd


def replace_number(x):
    if x == 1:
        return 1.0
    else:
        return -1.0

# init()
if __name__ == '__main__':
    load = LoadDatasets()
    X_train, y_train, X_test, y_test = load.load_mnist()
    y_train = y_train.apply(replace_number)
    y_test = y_test.apply(replace_number)
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)

    train_test(X_train, y_train, X_test, y_test)