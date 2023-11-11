import numpy as np
import random


class LocallyLinearSVMClassifier():
    # Locally Linear SVM with SGD

    def __init__(self, lamda, anchor_number, skip, t0, local_coding_model):
        self.lamda = lamda
        self.anchor_number = anchor_number
        self.skip = skip
        self.t0 = t0
        self.local_coding_model = local_coding_model

    def fit(self, X_train, y_train, epoch):
        feature_dim = X_train.shape[1]
        self.w = np.random.rand(self.anchor_number, feature_dim)
        self.b = np.random.rand(self.anchor_number, 1)
        count = self.skip
        for e in range(epoch):
            i = random.randint(0, X_train.shape[0] - 1)
            gamma = self.local_coding_model.fit(X_train[i])
            Ht = 1 - y_train[i] * (np.dot(np.dot(gamma.T, self.w), X_train[i]) + np.dot(gamma.T, self.b))
            if Ht > 0:
                self.w = self.w + 1 / (self.lamda * (e + self.t0)) * y_train[i] * (
                    np.dot(X_train[i].reshape(-1, 1), gamma.T)).T
                self.b = self.b + 1 / (self.lamda * (e + self.t0)) * y_train[i] * gamma
            count = count - 1
            if count <= 0:
                self.w = self.w * (1 - self.skip / (e + self.t0))
                count = self.skip

    def predict(self, test):
        gamma = self.local_coding_model.fit(test)
        return np.dot(np.dot(gamme.T, self.w), test) + np.dot(gamma.T, self.b)
