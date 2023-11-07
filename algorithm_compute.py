import math

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import time

from core_algorithm import LocallyLinearSVMClassifier, train_local_coding

def train_test(X_train, y_train, X_test, y_test, anchor_number):
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)

    labels = np.unique(y_train)
    model = train_local_coding(X_train=X_train, anchor_number=anchor_number)

    # Training
    print("start_train")
    train_start_time = time.time()
    svms = []
    for i, v in enumerate(labels):
        y_train_each = np.where(y_train == v, 1, -1)
        svm = LocallyLinearSVMClassifier(lamda=0.01, anchor_number=anchor_number, skip=50, t0=10,
                                         local_coding_model=model)
        svm.fit(X_train, y_train_each, epoch=50000)
        svms.append(svm)
    train_end_time = time.time()
    print("Training time:" + str(train_end_time-train_start_time)+"s")

    # Testing
    print("start_test")
    test_start_time = time.time()
    predicts = None
    for i, v in enumerate(labels):
        predict = []
        for j in range(0, len(X_test)):
            predict.append(svms[i].predict(X_test[j])[0][0])
        predict = np.array(predict)
        predict_tmp = np.where(predict >= 0, 1, -1)
        y_test_tmp = np.where(y_test == v, 1, -1)
        print("For label: {}, precision: {}, f1-measure: {}".format(v, precision_score(y_test_tmp, predict_tmp),
                                                                    f1_score(y_test_tmp, predict_tmp)))

        if predicts is None:
            predicts = predict
        else:
            predicts = np.vstack((predicts, predict))

    indexs = predicts.argmax(axis=0)
    y_predict = []
    for i, v in enumerate(indexs):
        y_predict.append(labels[v])
    print("Total test accuracy: {}".format(accuracy_score(y_test, y_predict)))
    test_end_time = time.time()
    print("Testing time:" + str(test_end_time - test_start_time) + "s")
