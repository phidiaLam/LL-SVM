import asyncio
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.datasets import make_moons, make_circles
from sklearn.inspection import DecisionBoundaryDisplay
from core_algorithm import LocallyLinearSVMClassifier
from local_coding import LocalCoding
import concurrent.futures

from logging_conf import logger_config

logger = logger_config(log_path='log.log')

def train(param):
    X_train, y_train, anchor_number, label, epoch, lamda, skip, t0 = param
    logger.info("start:{}".format(label))
    y_train_each = np.where(y_train == label, 1, -1)
    local_coding = LocalCoding(X_train, y_train_each, anchor_number)
    svm = LocallyLinearSVMClassifier(lamda=0.001, anchor_number=anchor_number, skip=skip, t0=t0,
                                     local_coding_model=local_coding)
    svm.fit(X_train, y_train_each, epoch=epoch)
    logger.info("complete train label:" + str(label))

    return label, svm

def test(param):
    svm, X_test, y_test, v = param
    predict = []
    for j in range(0, len(X_test)):
        predict.append(svm.predict(X_test[j])[0][0])
    predict = np.array(predict)
    predict_tmp = np.where(predict >= 0, 1, -1)
    y_test_tmp = np.where(y_test == v, 1, -1)
    logger.info("For label: {}, precision: {}, f1-measure: {}".format(v, precision_score(y_test_tmp, predict_tmp),
                                                                f1_score(y_test_tmp, predict_tmp)))
    return predict, v


def train_test(X_train, y_train, X_test, y_test, anchor_number, epoch, lamda, skip, t0):
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)

    labels = np.unique(y_train)

    # Training
    logger.info("start_train")
    train_start_time = time.time()
    params_train = []
    for i, v in enumerate(labels):
        params_train.append((X_train, y_train, anchor_number, v, epoch, lamda, skip, t0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        train_results = list(executor.map(train, params_train))
    train_end_time = time.time()
    logger.info("Training time:" + str(train_end_time - train_start_time) + "s")

    map_train = {}
    for (label, svm) in train_results:
        map_train[label] = svm

    # Testing
    logger.info("start_test")
    test_start_time = time.time()
    predicts = None
    params_test = []
    for i, v in enumerate(labels):
        params_test.append((map_train[v], X_test, y_test, v))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        test_results = list(executor.map(test, params_test))

    map_test = {}
    for (predict, label) in test_results:
        map_test[label] = predict

    for i, v in enumerate(labels):
        if predicts is None:
            predicts = map_test[v]
        else:
            predicts = np.vstack((predicts, map_test[v]))

    indexs = predicts.argmax(axis=0)
    y_predict = []
    for i, v in enumerate(indexs):
        y_predict.append(labels[v])
    logger.info("Total test accuracy: {}".format(accuracy_score(y_test, y_predict)))
    test_end_time = time.time()
    logger.info("Testing time:" + str(test_end_time - test_start_time) + "s")


def moon_train(anchor_number):
    dataset = make_circles(n_samples=1000, noise=0.05)

    X, y = dataset[0], dataset[1]
    y = np.where(y == 1, 1, -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    feature_1, feature_2 = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max()),
        np.linspace(X[:, 1].min(), X[:, 1].max())
    )

    grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

    local_coding = LocalCoding(X_train, y_train, anchor_number)
    anchors = local_coding.get_anchor()
    svm = LocallyLinearSVMClassifier(lamda=0.001, anchor_number=anchor_number, skip=10, t0=10,
                                     local_coding_model=local_coding)
    svm.fit(X_train, y_train, epoch=500000)
    predict = []
    for j in range(0, len(grid)):
        predict.append(svm.predict(grid[j])[0][0])
    predict = np.array(predict)
    predict = np.where(predict >= 0, 1, -1)
    y_pred = np.reshape(predict, feature_1.shape)

    display = DecisionBoundaryDisplay(
        xx0=feature_1, xx1=feature_2, response=y_pred
    )
    display.plot()
    display.ax_.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="black"
    )
    display.ax_.scatter(
        anchors[:, 0], anchors[:, 1], c='green', edgecolor="pink"
    )
    plt.title('ll-svm')
    plt.show()

    predict = []
    for j in range(0, len(X_test)):
        predict.append(svm.predict(X_test[j])[0][0])
    predict = np.array(predict)
    predict = np.where(predict >= 0, 1, -1)

    print("accuracy is {}".format(accuracy_score(y_test, predict)))

def banana_train(anchor_number):
    dataset = pd.read_csv("./datasets/banana/banana.csv",sep=",")

    X, y = dataset[['At1','At2']].values, dataset.Class.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    feature_1, feature_2 = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max()),
        np.linspace(X[:, 1].min(), X[:, 1].max())
    )

    grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T

    local_coding = LocalCoding(X_train, y_train, anchor_number)
    anchors = local_coding.get_anchor()
    svm = LocallyLinearSVMClassifier(lamda=0.001, anchor_number=anchor_number, skip=10, t0=10,
                                     local_coding_model=local_coding)
    svm.fit(X_train, y_train, epoch=500000)
    predict = []
    for j in range(0, len(grid)):
        predict.append(svm.predict(grid[j])[0][0])
    predict = np.array(predict)
    predict = np.where(predict >= 0, 1, -1)
    y_pred = np.reshape(predict, feature_1.shape)

    display = DecisionBoundaryDisplay(
        xx0=feature_1, xx1=feature_2, response=y_pred
    )
    display.plot()
    display.ax_.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, edgecolor="black"
    )
    display.ax_.scatter(
        anchors[:, 0], anchors[:, 1], c='green', edgecolor="pink"
    )
    plt.title('ll-svm')
    plt.show()

    predict = []
    for j in range(0, len(X_test)):
        predict.append(svm.predict(X_test[j])[0][0])
    predict = np.array(predict)
    predict = np.where(predict >= 0, 1, -1)

    print("accuracy is {}".format(accuracy_score(y_test, predict)))

