import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt


class LinearSVMClassifier():
    # Linear SVM with SGD
    # learning rate decay: lr = m/(k+n), k = number of seasons, m = lr_numerator_factor, n = lr_denominator_factor

    def __init__(self, batch_size, lamda, lr_numerator_factor, lr_denominator_factor, step_size):
        self.batch_size = batch_size
        self.lamda = lamda
        self.lr_numerator_factor = lr_numerator_factor
        self.lr_denominator_factor = lr_denominator_factor
        self.step_size = step_size

    def batch(self, size):
        tmp = np.array(range(size))
        np.random.shuffle(tmp)
        # Returns an list of index array, each array in the list represents the set of data records in one batch
        # if the remainder less than batch_size, then the last batch will just be the remaining data records
        return [tmp[i * self.batch_size:min((i + 1) * self.batch_size, size)] for i in
                range(int(np.ceil(size / self.batch_size)))]

    def compute_cost(self, X, Y):
        # calculate hinge loss
        N = X.shape[0]
        distances = 1 - Y * (np.dot(X, self.w)+self.b)
        distances[distances < 0] = 0  # equivalent to max(0, distance)
        hinge_loss = 1000 * (np.sum(distances) / N)

        # calculate cost
        cost = 1 / 2 * np.dot(self.w, self.w) + hinge_loss
        return cost

    def fit_eval(self, train, train_label, epoch, val=None, val_label=None):
        # If validation set provided, then validation set is used for tune parameters and report accuracy or other metrics
        # If no validation set provided, then set aside random 50 samples for heldout set for each epoch
        # The heldout set is used for tune parameters and report accuracy or other metrics
        # train, val: training/validation set features, of dim (n_samples, n_features)
        # train_label, val_label: training/validation set labels, of dim (n_samples, )
        accuracy = []
        coef_size = []
        feature_dim = train.shape[1]
        self.w = np.random.rand(feature_dim)
        self.b = 0
        # If validation set provided
        if ((val is not None) and (val_label is not None)):
            X_train = train
            y_train = train_label
            X_val = val
            y_val = val_label

        print_number = 0
        pre_cost = 0
        for e in range(epoch):
            # If no validation set provided
            if ((val is None) or (val_label is None)):
                # Random select 50 for each epoch as held out set
                tmp = np.array(range(len(train_label)))
                np.random.shuffle(tmp)

                # Train set for the epoch
                X_train = train[tmp[:-50]]
                y_train = train_label[tmp[:-50]]

                # Held out set for the epoch
                X_val = train[tmp[-50:]]
                y_val = train_label[tmp[-50:]]

            batch_idx = self.batch(len(y_train))

            for i in range(len(batch_idx)):
                # Implementation of learning rate decay, Learning rate = m / (n+k), m, n -> hyper parameter, k -> # of season
                lr_numseason_factor = (e * np.ceil(len(batch_idx) / self.step_size) + np.floor(i / self.step_size))
                learning_rate = self.lr_numerator_factor / (self.lr_denominator_factor + lr_numseason_factor)

                gradient_w = np.sum(np.expand_dims(
                    np.where((y_train[batch_idx[i]] * (np.sum(self.w * X_train[batch_idx[i]], axis=1) + self.b)) > 1, 0,
                             -1), 1)
                                    * np.expand_dims(y_train[batch_idx[i]], 1) * X_train[batch_idx[i]], axis=0) / len(
                    batch_idx[i]) + self.lamda * self.w
                gradient_b = np.sum(
                    np.where((y_train[batch_idx[i]] * (np.sum(self.w * X_train[batch_idx[i]], axis=1) + self.b)) > 1, 0,
                             -1)
                    * y_train[batch_idx[i]]) / len(batch_idx[i])

                self.w = self.w - learning_rate * gradient_w
                self.b = self.b - learning_rate * gradient_b

                if (i % self.step_size == 0):
                    accuracy.append(self.score(X_val, y_val))
                    coef_size.append(np.sum(np.square(self.w)))

            if 2 ** print_number == e:
                cost = self.compute_cost(train, train_label)
                if cost != 0 and (pre_cost - cost)**2 < 0.25:
                    break
                print("Epoch is: {} and Cost is: {}".format(e, cost))
                pre_cost = cost
                print_number += 1

        return accuracy, coef_size

    def predict(self, test):
        return np.where((np.sum(self.w * test, axis=1) + self.b) >= 0, 1, -1)

    def score(self, test, test_label):
        # test: test set features, of dim (n_samples, n_features)
        # test_label: test set labels, of dim (n_samples, )
        y_predict = self.predict(test)
        return np.mean(y_predict == test_label)


def init():
    print("reading dataset...")
    # read data in pandas (pd) data frame
    data = pd.read_csv('./data/data.csv')
    print(data)
    # drop last column (extra column added by pd)
    # and unnecessary first column (id)
    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
    print(data)
    print("applying feature engineering...")
    # convert categorical labels to numbers
    diag_map = {'M': 1.0, 'B': -1.0}
    print(data['diagnosis'])
    data['diagnosis'] = data['diagnosis'].map(diag_map)
    print(data)
    # put features & outputs in different data frames
    Y = data.loc[:, 'diagnosis']
    X = data.iloc[:, 1:]
    X = X.values
    Y = Y.values
    print(Y)
    print(X)

    # normalize data for better convergence and to prevent overflow
    X = MinMaxScaler().fit_transform(X)

    # split data into train and test set
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)
    svm = LinearSVMClassifier(batch_size=128, lamda=0.00001, lr_numerator_factor=1, lr_denominator_factor=100,
                              step_size=30)
    svm.fit_eval(X_train, y_train, epoch=500)
    y_predict = svm.predict(X_test)

    print("accuracy on test dataset: {}".format(accuracy_score(y_test, y_predict)))
    print("recall on test dataset: {}".format(recall_score(y_test, y_predict)))
    print("precision on test dataset: {}".format(precision_score(y_test, y_predict)))

#
def train_test(X_train, y_train, X_test, y_test):
    # normalize data for better convergence and to prevent overflow
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)
    # train the model
    print("training started...")

    svm = LinearSVMClassifier(batch_size=128, lamda=0.00001, lr_numerator_factor=1, lr_denominator_factor=100,
                              step_size=30)
    svm.fit_eval(X_train, y_train, epoch=500)
    print("training finished.")
    y_predict = svm.predict(X_test)

    print("accuracy on test dataset: {}".format(accuracy_score(y_test, y_predict)))
    print("recall on test dataset: {}".format(recall_score(y_test, y_predict)))
    print("precision on test dataset: {}".format(precision_score(y_test, y_predict)))

    position_index = []
    for i, v in enumerate(y_predict):
        if v == 1:
            position_index.append(i)

    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i, v in enumerate(position_index):
        images = np.reshape(X_test[v], [16,16])
        ax = fig.add_subplot(6, 6, i + 1, xticks=[], yticks=[])
        ax.imshow(images, cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(y_test[i]))
        if i+1 >= 36:
            break
    plt.show()
