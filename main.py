import asyncio
import sys

from algorithm_compute import train_test, moon_train
from load_datasets import LoadDatasets
import numpy as np

from logging_conf import logger_config

if __name__ == '__main__':
    load = LoadDatasets()
    # dataset = sys.argv[1]
    # anchor_number = int(sys.argv[2])
    # if type(anchor_number)!=int or anchor_number<=0:
    #     print("Anchor Number need integer")
    #     exit(1)
    # print("start load datasets")
    # X_train, y_train, X_test, y_test = None, None, None, None
    # match dataset:
    #     case 'mnist':
    #         X_train, y_train, X_test, y_test = load.load_mnist()
    #     case 'usps':
    #         X_train, y_train, X_test, y_test = load.load_usps()
    #     case 'letter':
    #         X_train, y_train, X_test, y_test = load.load_letter()
    #     case 'caltech':
    #         X_train, y_train, X_test, y_test = load.load_caltech()
    #     case 'moon':
    #         moon_train(anchor_number)
    #         exit(0)
    #     case _:
    #         print("Can not found dataset")
    #         exit(1)
    # train_test(X_train, y_train, X_test, y_test, anchor_number)
    X_train, y_train, X_test, y_test = load.load_mnist()
    epochs = [1000, 10000, 100000, 1000000]
    lamdas = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    archors = [10, 30, 50, 70, 100]
    skip_t0s = [(5, 5), (10, 10), (50, 10), (10, 50), (50, 50), (100, 100)]
    datasets = [load.load_mnist(), load.load_usps(), load.load_letter(), load.load_caltech()]
    datasets_name = ["mninst", "usps", "letter", "caltech"]
    for i, v in enumerate(datasets):
        (X_train, y_train, X_test, y_test) = v
        for epoch in epochs:
            for lamda in lamdas:
                for archor in archors:
                    for skip, t0 in skip_t0s:
                        train_test(X_train, y_train, X_test, y_test, archor, epoch, lamda, skip, t0, datasets_name[i])


