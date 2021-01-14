from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def rbfGaussiana(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)


def computeEqualStds(centers, k):
    dist = [np.sqrt(np.sum((c1 - c2) ** 2))
            for c1 in centers for c2 in centers]
    dMax = np.max(dist)
    stds = np.repeat(dMax / np.sqrt(2 * k), k)
    return stds


class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k=3, attnumber=4, lr=0.01, epochs=100, rbf=rbfGaussiana, computeStds=computeEqualStds):
        self.k = k  # grupos ou numero de neuronios na camada escondida
        self.lr = lr  # taxa de aprendizagem
        self.epochs = epochs  # número de iterações
        self.rbf = rbf  # função de base radial
        self.computeStds = computeStds  # função de cálculo da largura do campo receptivo
        self.w = np.random.randn(self.k, attnumber)
        self.b = np.random.randn(1)

    def fit(self, X, y):
        self.stds = []
        # K-Means pra pegar os centros inicias
        # 1º parâmetro da rede RBF
        kmeans = KMeans(
            n_clusters=self.k, init='random',
            n_init=10, max_iter=300).fit(X)
        self.centers = kmeans.cluster_centers_
        #print('centers: ', self.centers)

        # Cálculo da largura do campo receptivo
        # 2º parâmetro da rede RBF
        self.stds = self.computeStds(self.centers, self.k)
        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                # calcula a saída de cada neurônio da função de base radial
                phi = np.array([self.rbf(X[i], c, s)
                                for c, s, in zip(self.centers, self.stds)])
                # calcula somatório do produto da saída da função de base radial e os pesos
                F = phi.T.dot(self.w)
                F = np.sum(F) + self.b
                # saída da rede
                out = 0 if F < 0 else 1
                # função de perda
                loss = (y[i] - out).flatten() ** 2
                #print('Loss: {0:.2f}'.format(loss[0]))
                # cálculo do erro
                error = (y[i] - out).flatten()
                # atualização dos pesos
                # 3º Parâmetro da rede
                self.w = self.w + self.lr * error * phi
                self.b = self.b + self.lr * error

    # calcula saída da rede RBF com a rede treinada
    def predict(self, X):
        y_pred = []
        error = 0
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s)
                          for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w)
            F = np.sum(F) + self.b
            out = 0 if F < 0 else 1
            y_pred.append(out)
        return np.array(y_pred)


def load_data():
    url = 'moncattle/data/lomba.csv'
    df = pd.read_csv(url)
    # remove a ultima coluna (dados)
    #data = df[df.columns[1:10]]
    data = df[df.columns[1:4]]
    # normaliza os dados
    normalized_data = (data - data.min()) / (data.max() - data.min())
    # retorna a última coluna (rótulos)
    labels = df[df.columns[-1]]
    # separa em conjunto de treinamento e teste com seus respectivos rótulos
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_data, labels, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


# chama função que carrega base de dados
training_inputs, test_inputs, training_labels, test_labels = load_data()
# transforma rótulos do conjunto de treinamento em numeros pra calculo do erro
le = preprocessing.LabelEncoder()
le.fit(training_labels.values)
training_labels_transformed = le.transform(training_labels.values)
# chama RBF
rbfnet = RBFNet(attnumber=3, k=5, computeStds=computeEqualStds)
rbfnet.fit(training_inputs.values, training_labels_transformed)
# transforma rótulos do conjunto de teste em numeros pra calculo do erro
le = preprocessing.LabelEncoder()
le.fit(test_labels.values)
test_labels_transformed = le.transform(test_labels.values)
#y_pred = rbfnet.predict(test_labels_transformed)
y_pred = rbfnet.predict(test_inputs.values)
errorabs = sum(abs(test_labels_transformed-y_pred) == 0)
print('error: ', np.sum(errorabs)/len(test_labels_transformed))
