import operator
import warnings

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


warnings.filterwarnings("ignore")


class Score():
    def __init__(self, score, mean, std, learning_rate, l2, momentum):
        self.value = score
        self.mean = mean
        self.std = std
        self.learning_rate = learning_rate
        self.l2 = l2
        self.momentum = momentum

    def __repr__(self):
        return "Score: {}, Mean: {}, Std: {}, Learning Rate: {}, L2 (regularization term): {}, Momentum: {}".format(
            self.value, self.mean, self.std, self.learning_rate, self.momentum, self.l2)


if __name__ == "__main__":
    df = pd.read_csv('moncattle/data/lomba.csv')
    data = df[df.columns[1:10]]
    data = (data - data.min()) / (data.max() - data.min())
    labels = df[df.columns[-1]]
    le = preprocessing.LabelEncoder()
    le.fit(labels.values)
    labels = le.transform(labels.values)
    learning_rate = 0.3
    values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    outputs = []
    for learning_rate in values:
        for l2 in values:
            for momentum in values:
                print("Learning Rate: {}, L2 (regularization term): {}, Momentum: {}".format(
                    learning_rate, l2, momentum))
                clf = MLPClassifier(solver='sgd', activation='logistic',
                                    hidden_layer_sizes=(9, 6), batch_size=24,
                                    learning_rate_init=learning_rate, max_iter=20,
                                    momentum=momentum, alpha=l2, shuffle=True)
                scores = cross_val_score(clf, data, labels, cv=10)
                # [print("{}-Fold: {:.2f}".format(k, score))
                # for k, score in enumerate(scores, 1)]
                mean = np.mean(scores)
                std = np.std(scores)
                score = (mean/std)
                outputs.append(
                    Score(score, mean, std, learning_rate, l2, momentum))
                print("\tAcurácia média: {:.2f}".format(mean))
                print("\tDesvio Padrão: {:}".format(std))
                print("\tScore: {}".format(score))
    best = sorted(outputs, key=operator.attrgetter("value"))
    print(best)
