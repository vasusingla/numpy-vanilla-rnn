import numpy as np

## TODO - not implementing activation for now, since just gonna try one layer rnn first & criterion computes with logits
## can maybe implement an activation layer seperately later?
class FullyConnected:

    def __init__(self, input_dimension, output_dimension):

        self._W = np.random.randn(input_dimension, output_dimension) * 0.01
        self._b = np.zeros((output_dimension, 1))
        self._dW = np.zeros(self._W.shape)
        self._db = np.zeros(self._b.shape)

    def forward(self, X):
        '''
        :param X: (input_dimension, batch_size)
        :return: Z (output_dimension, batch_size)
        '''
        # self._X = X
        self._Z = np.dot(self._W.T, X) + self._b
        return self._Z

    def backward(self, X, gradOut):
        '''
        :param X: (input_dimension, batch_size)
        :param gradOut: (output_dimension, batch_size)
        :return:
        '''
        m = X.shape[1]
        self._db = (1./m)*np.sum(gradOut, axis=0, keepdims=True).T
        self._dW = (1./m)*np.dot(X, gradOut)
        dX = np.dot(self._W, gradOut.T)
        assert dX.shape==X.shape
        assert self._db.shape==self._b.shape
        assert self._W.shape==self._dW.shape
        return dX

    def update_parameters(self, learning_rate=0.01):
        self._W -= self._dW*learning_rate
        self._b -= self._db*learning_rate
        self._dW[:, :] = 0
        self._db[:, :] = 0
