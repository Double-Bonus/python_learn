# Tinklas from scrath
import numpy as np
from numpy.lib.function_base import select


class neural_network(object):
    def __init__(self):
        #parametrai tinklo strukturai
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # atsitiktiniai koeficientai
        self.W1 = np.random.rand(self.inputSize,self.hiddenSize)   # matrica 2x3
        self.W2 = np.random.randn(self.hiddenSize,self.outputSize) # 3x1

    # neurono aktyvacijos funkcija
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    # feedforward
    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o


    def sigmoidPrime(self, s):
        return s * (1-s)

    # backpropagation
    def backward(self, X, y, output):
        self.o_error = y - output
        self.o_delta = self.o_error * self.sigmoidPrime(output)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

        # svoriu pakeitimas
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X,y,o)


    def predict(self, X):
        # print("Suprognuozota verte: {}".format(self.forward(X)))
        return self.forward(X)

        
# visi duomenys
# col1 = miego valandos, col2 = laikas skirtas studijoms, col3 = testo rezultatas

x_all = np.array([
    [9, 2, 80],
    [8, 4, 90],
    [5, 1, 51],
    [5, 10, 95]
], dtype=np.float)

X = x_all[:,0:2]
X[:,0] = X[:,0]/12.0
X[:,1] = X[:,1]/12.0

y = np.array([[80],[90],[51],[95]], dtype=np.float)
y = y / 100.0
# y = x_all[:,2]/100.0

# print(x_all)
print(X)

nn = neural_network()

for i in range(0,100):
    print("Loss {}".format(np.mean(np.sqrt( (y - nn.forward(X))**2 ))))
    nn.train(X,y)
    


xtest = np.array([8,3],dtype=float)
# koks tinklo rezultatas

print("Tiketina kad gausi: {}".format(nn.predict(xtest)*100))
