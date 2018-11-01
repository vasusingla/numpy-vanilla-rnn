import numpy as np


class Embedding:

    def __init__(self, embedding_size, vocab_size):
        self._embedding_size = embedding_size
        self._ematrix = np.random.normal(size = (vocab_size, embedding_size))
        self._dematrix = np.zeros((vocab_size, embedding_size))

    def forward(self, X):
        '''
        :param X: (batch_size, sequence_length)
        :return: (embedding_size, batch_size, sequence_length)
        '''
        embed_X = np.zeros((self._embedding_size, X.shape[0], X.shape[1])).astype(np.float64)
        for i in range(X.shape[0]):
            # embed_X[:, i, :] = self._ematrix[X[i, :]]
            embed_X[:,i,:] = np.take(self._ematrix, X[i,:], axis=0).T.astype(np.float64)
        return embed_X

    def backward(self, gradOut, X):
        '''
        :param gradOut: (embedding_size, batch_size, sequence_length)
        :param X: (batch_size, sequence_length)
        :return: None
        '''
        # Tried to think of a more vectorized implementation of this, but it isn't possible.
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                idx = X[i,j]
                self._dematrix[idx, :]+=gradOut[:,i,j]

    def update_params(self, learning_rate=0.01):
        self._ematrix -= learning_rate*self._dematrix
        self._dematrix=np.zeros(self._dematrix.shape)
        return
