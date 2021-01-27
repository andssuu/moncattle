from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    df = pd.read_csv('moncattle/data/lomba.csv')
    data = df[df.columns[1:10]]
    data = (data - data.min()) / (data.max() - data.min())
    labels = df[df.columns[-1]]
    le = preprocessing.LabelEncoder()
    le.fit(labels.values)
    labels = le.transform(labels.values)
    learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    for learning_rate in learning_rates:
        print("Learning Rate --> {}".format(learning_rate))
        clf = MLPClassifier(solver='sgd', activation='logistic',
                            hidden_layer_sizes=(9, 6), batch_size=24,
                            learning_rate_init=learning_rate, shuffle=True)
        scores = cross_val_score(clf, data, labels, cv=10)
        [print("{}-Fold: {:.2f}".format(k, score))
         for k, score in enumerate(scores, 1)]
        print("Acurácia média: {:.2f}".format(np.mean(scores)))
