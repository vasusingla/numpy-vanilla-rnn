from RNN import RNN
from FullyConnected import FullyConnected
from Embedding import Embedding
import numpy as np

# Implementing only for 2 class, can be easily updated, requires modification to init


class Model:

    def __init__(self, nLayers, H, V, E):
        """
        :param nLayers: number of RNN layers
        :param H: Hidden size dimension of RNN
        :param V: Vocabulary size
        :param E: embedding size
        """
        self._nLayers = nLayers
        self._H = H
        self._V = V
        self._E = E
        assert self._nLayers >= 1
        self._rnn_list = [RNN(E, H)]
        self._input_list = []
        for i in (range(self._nLayers-1)):
            self._rnn_list.append(RNN(H, H))
        self._embed = Embedding(E, V)
        # change num classes here
        self._fc_last = FullyConnected(H, 2)

    def forward(self, X):
        """
        :param X: (batch_size, sequence_length)
        :return: (batch_size, 2)
        """
        embed_X = self._embed.forward(X)
        input_data = embed_X
        self._input_list = []
        for i in xrange(self._nLayers):
            self._input_list.append(input_data)
            input_data = self._rnn_list[i].forward(input_data)
        self._input_list.append(input_data)
        # Fully connected takes last thought output as input
        self._fc_last_in = input_data[:, :, -1]
        out_data = self._fc_last.forward(self._fc_last_in)

        # FC return (2, batch_size)
        return out_data.T

    def backward(self, X, gradOutput):
        """
        :param X: (batch_size, sequence_length)
        :param gradOutput: (batch_size, 2)
        :return: None
        """
        # FC requires (2, batch_size)
        gradOut_last = self._fc_last.backward(self._fc_last_in, gradOutput)
        gradOut_t = np.zeros((self._H, X.shape[0], X.shape[1]))
        gradOut_t[:,:, -1] = gradOut_last
        for i in reversed(range(1, self._nLayers)):
            gradOut_t = self._rnn_list[i].backward(self._input_list[i], gradOut_t, self._input_list[i+1])
        gradOut_t = self._rnn_list[0].backward(self._input_list[0], gradOut_t, self._input_list[1])
        self._embed.backward(gradOut_t, X)

    def update_parameters(self, learning_rate = 0.01):
        self._fc_last.update_parameters(learning_rate)
        for i in range(self._nLayers):
            self._rnn_list[i].update_params(learning_rate)
        self._embed.update_params(learning_rate)
