import pickle
import gzip
import random
import numpy as np


# Função de Ativação Sigmóide
def sigmoid(net):
    return 1.0/(1.0+np.exp(-net))

# Função para retornar as derivadas da função Sigmóide


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

# Classe Network


class Network(object):

    def __init__(self, sizes, activation_function=sigmoid,
                 prime_function=sigmoid_prime):
        self.num_layers = len(sizes)  # número de neurônios em cada camada
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]  # limiar
        self.weights = [np.random.randn(y, x) for x, y in zip(
            sizes[:-1], sizes[1:])]  # pesos
        self.activation_function = activation_function
        self.prime_function = prime_function

    def feedforward(self, x):
        """Retorna a saída da rede z se `x` for entrada."""
        for b, w in zip(self.biases, self.weights):
            x = self.activation_function(np.dot(w, x)+b)  # net = (∑xw+b)
        return x

    def SGD(self, training_data, epochs, mini_batch_size, _n, test_data=None):
        # dataset de treino
        training_data = list(training_data)
        n = len(training_data)
        # dataset de teste
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
        for j in range(epochs):
            random.shuffle(training_data)
            # técnica que realiza o treinamento por lotes
            # mini_batch_size = tamanho do lote
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, _n)
            if test_data:
                acc = self.evaluate(test_data)
                print("Epoch {} : {} / {} = {}%".format(j,
                                                        acc, n_test, (acc*100)/n_test))
            else:
                print("Epoch {} finalizada".format(j))

    def update_mini_batch(self, mini_batch, _n):
        # inicializa matriz com derivadas de pesos e limiares
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            # resultado dos deltas do backpropagation sem a multiplicação da taxa de aprendizagem
            # soma os deltas do minibatch
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        # atualiza pesos e limiares (𝜂*𝛿*f’(net)*𝑥)
        self.weights = [w-(_n/len(mini_batch))*nw for w,
                        nw in zip(self.weights, nabla_w)]
        self.biases = [b-(_n/len(mini_batch))*nb for b,
                       nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Retorna uma tupla `(nabla_b, nabla_w)` representando o
         gradiente para a função de custo J_x. `nabla_b` e
         `nabla_w` são listas de camadas de matrizes numpy, semelhantes
         a `self.biases` e `self.weights`."""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # Feedforward
        activation = x
        # Lista para armazenar todas as saídas dos neurônios (z), camada por camada
        activations = [x]
        # Lista para armazenar todos os vetores net, camada por camada
        nets = []
        for b, w in zip(self.biases, self.weights):
            net = np.dot(w, activation)+b
            nets.append(net)
            activation = self.activation_function(
                net)  # z = valor de saída do neurônio
            activations.append(activation)
        # Backward pass
        # última camada -(u-z)f'(net)
        delta = self.cost_derivative(
            activations[-1], y) * self.prime_function(nets[-1])
        nabla_b[-1] = delta
        # (𝑦−𝑧)*f’(net)*𝑥
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # l = 1 significa a última camada de neurônios, l = 2 é a penúltima e assim por diante.
        for l in range(2, self.num_layers):
            net = nets[-l]
            zs = self.prime_function(net)
            # delta da camada intermediaria. Note que utiliza o delta calculado anteriormente
            delta = np.dot(self.weights[-l+1].transpose(), delta) * zs
            nabla_b[-l] = delta
            # ∑(𝛿𝑤)f’(net)𝑥
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Retorna o número de entradas de teste para as quais a rede neural 
         produz o resultado correto. Note que a saída da rede neural
         é considerada o índice de qualquer que seja
         neurônio na camada final que tenha a maior ativação."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Retorna o vetor das derivadas parciais."""
        return (output_activations-y)


def load_data():
    f = gzip.open('redes_neurais_pos/MLP/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(
        f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()
    training_data = list(training_data)

    # arquitetura da rede
    arquitecture = [784, 30, 20, 10]
    mlp = Network(arquitecture)
    mlp.SGD(training_data, 10, 32, 0.3, test_data=test_data)
