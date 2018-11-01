import numpy as np


def one_hot_convert(target, n):
    m = target.shape[0]
    target_one_hot = np.zeros((m, n))
    target_one_hot[np.arange(m), target] = 1
    return target_one_hot


class Criterion:

    def _softmax(self, x):
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=1, keepdims=True)

    def forward(self, input, target):
        '''
        :param input: (batch_size, num_of_classes)
        :param target: (batch_size)
        :return: scalar
        '''
        target_one_hot = one_hot_convert(target, input.shape[1])
        X = self._softmax(input)
        N = target.shape[0]
        cross_entropy = -np.sum(target_one_hot * np.log(X + 1e-9)) / N
        return cross_entropy

    def backward(self, input, target):
        '''
        :param input: (batch_size, num_of_classes)
        :param target: (batch_size)
        :return: (batch_size, num_of_classes)
        '''
        m = input.shape[0]
        grad = self._softmax(input)
        grad[range(m), target] -= 1
        grad = grad / m
        return grad
