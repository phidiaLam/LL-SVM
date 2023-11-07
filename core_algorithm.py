import pandas as pd
import numpy as np
from mlpack import local_coordinate_coding
import random


def train_local_coding(X_train, anchor_number):
    output = local_coordinate_coding(training=X_train, atoms=anchor_number, lambda_=0.0001, max_iterations=10)
    return output['output_model']

class LocallyLinearSVMClassifier():
    # Locally Linear SVM with SGD
    # learning rate decay: lr = m/(k+n), k = number of seasons, m = lr_numerator_factor, n = lr_denominator_factor

    def __init__(self, lamda, anchor_number, skip, t0, local_coding_model):
        self.lamda = lamda
        self.anchor_number = anchor_number
        self.skip = skip
        self.t0 = t0
        self.local_coding_model = local_coding_model
    def use_local_coding(self, data):
        result = local_coordinate_coding(input_model=self.local_coding_model, test=data)
        coding = result['codes']
        coding = coding / np.sum(coding)
        return coding.T

    def fit(self, X_train, y_train, epoch):
        feature_dim = X_train.shape[1]
        self.w = np.random.rand(self.anchor_number, feature_dim)
        self.b = np.random.rand(self.anchor_number, 1)
        count = self.skip
        for e in range(epoch):
            i = random.randint(0,X_train.shape[0]-1)
            gamme = self.use_local_coding([X_train[i]])
            Ht = 1 - y_train[i] * (np.dot(np.dot(gamme.T,self.w),X_train[i])+np.dot(gamme.T,self.b))
            if Ht > 0:
                self.w = self.w + 1 / (self.lamda * (e + self.t0)) * y_train[i] * (np.dot(X_train[i].reshape(-1,1),gamme.T)).T
                self.b = self.b + 1 / (self.lamda * (e + self.t0)) * y_train[i] * gamme
            count = count - 1
            if count <= 0:
                self.w = self.w*(1-self.skip / (e + self.t0))
                count = self.skip

    def predict(self, test):
        gamme = self.use_local_coding([test])
        return np.dot(np.dot(gamme.T,self.w),test)+np.dot(gamme.T,self.b)
