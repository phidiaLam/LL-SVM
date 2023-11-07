import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler

from core_algorithm import LocallyLinearSVMClassifier, train_local_coding

def train_test(X_train, y_train, X_test, y_test, anchor_number):
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)

    labels = np.unique(y_train)
    predicts = None
    model = train_local_coding(X_train=X_train, anchor_number=anchor_number)
    print("start_train")
    for i, v in enumerate(labels):
        y_train_each = np.where(y_train == v, 1, -1)
        svm = LocallyLinearSVMClassifier(lamda=0.01, anchor_number=anchor_number, skip=50, t0=10,
                                         local_coding_model=model)
        svm.fit(X_train, y_train_each, epoch=5000)
        predict = []
        for j in range(0, len(X_test)):
            predict.append(svm.predict(X_test[j])[0][0])
        predict = np.array(predict)
        # predict = svm.predict(X_test)
        predict_tmp = np.where(predict >= 0, 1, -1)
        y_test_tmp = np.where(y_test == v, 1, -1)
        print(precision_score(y_test_tmp, predict_tmp))

        position_index = []
        for i, v in enumerate(predict_tmp):
            if v == 1:
                position_index.append(i)

        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        for i, v in enumerate(position_index):
            images = np.reshape(X_test[v], [int(math.sqrt(len(X_test[v]))), int(math.sqrt(len(X_test[v])))])
            ax = fig.add_subplot(6, 6, i + 1, xticks=[], yticks=[])
            ax.imshow(images, cmap=plt.cm.binary, interpolation='nearest')
            ax.text(0, 7, str(y_test[v]))
            if i + 1 >= 36:
                break
        plt.show()

        if predicts is None:
            predicts = predict
        else:
            predicts = np.vstack((predicts, predict))

    indexs = predicts.argmax(axis=0)
    y_predict = []
    for i, v in enumerate(indexs):
        y_predict.append(labels[v])
    print(accuracy_score(y_test, y_predict))
