import numpy as np
# TODO - bias output gradient check


class RNN:

    def __init__(self, input_dimension, hidden_dimension,  init_type='xavier', act_type='tanh'):
        # Gonna keep Xavier Initialization as default, for now and activation as tanh
        if init_type=='xavier':
            self._Wa = np.random.uniform(-np.sqrt(1./hidden_dimension), np.sqrt(1./hidden_dimension),
                                      size=(input_dimension+hidden_dimension, hidden_dimension))
            # self._Wo = np.random.uniform(-np.sqrt(1./hidden_dimension), np.sqrt(1./hidden_dimension),
            #                           size=(hidden_dimension, input_dimension))
        else:
            raise NotImplementedError("Initialization not implemented")
        # self._batch_size = batch_size
        self._ba = np.random.random(size=(hidden_dimension, 1))
        # self._bo = np.zeros(shape=(input_dimension, 1))
        self._dWa = np.zeros(shape=self._Wa.shape)
        self._dba = np.zeros(shape=self._ba.shape)
        # self._dWo = np.zeros(shape=self._Wo.shape)
        # self._dbo = np.zeros(shape=self._bo.shape)
        self._input_dimension = input_dimension
        self._hidden_dimension = hidden_dimension
        # self._a = np.zeros(shape=(hidden_dimension, 1))
        if act_type not in ('tanh'):
            raise NotImplementedError("Unimplemented activation type")
        self._act_type = act_type

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def _sigmoid_d(self, x):
        return x

    def _tanh(self, x):
        return np.tanh(x)

    def _tanh_d(self, x):
        return(1 - x**2)


    def _step_forward(self, in_t, a_prev):
        '''
        :param in_t: time-step input, (input_dimension, batch_size)
        :param a_prev: previous hidden state at t-1, (hidden_size, batch_size)
        :return: current hidden state at t
        '''
        concat_mat = np.concatenate((in_t, a_prev))
        if self._act_type=='tanh':
            a_next = self._tanh(np.dot(self._Wa.T, concat_mat) + self._ba)
            # o = (np.dot(self._Wo.T, a_next))
        else:
            raise NotImplementedError("Type of activation has not been defined")
        return a_next

    def forward(self, input, a0 = None):
        """
        :param input: complete input size, (input_dimension, batch_size, sequence_length)
        :return: final activation, (hidden_dimension. batch_size, sequence_length)
        """
        T = input.shape[-1]
        a_arr = np.zeros((self._hidden_dimension, input.shape[1], T))
        # o_arr = np.zeros((self._input_dimension, input.shape[1], T))
        if a0 is None:
            a0 = a_arr[:,:,0]
        a_next = a0
        self._a0 = a0
        for t in xrange(T):
            a_arr[:, :, t] = self._step_forward(input[:,:,t], a_next)
            a_next = a_arr[:, :, t]
        return a_arr

    def _step_backward(self, da_next, a_next, a_prev, in_t):
        dtanh = self._tanh_d(a_next)*da_next
        Win_t = self._Wa[:self._input_dimension, :]
        Waa = self._Wa[self._input_dimension:, :]

        din_t = np.dot(Win_t, dtanh)
        da_prev = np.dot(Waa, dtanh)
        concat_mat = np.concatenate((in_t, a_prev))
        dWa_t = np.dot(dtanh, concat_mat.T).T
        dba_t = np.sum(dtanh, axis=1, keepdims=True)

        return din_t, da_prev, dWa_t, dba_t

    def backward(self, input, gradOutput, backprop_cache):
        """
        :param input: complete input size, (input_dimension, batch_size, sequence_length)
        :param gradOutput: (hidden_dimension, batch_size, sequence_length)
        :param backprop_cache: [hidden_state_arr]
        :return gradInput: same dim as input
        """
        a_arr = backprop_cache
        din = np.zeros(shape=input.shape)
        da_prev_t = np.zeros((self._hidden_dimension, input.shape[1]))
        for i in reversed(range(input.shape[-1])):
            if i!=0:
                din_t, da_prev_t, dWa_t, dba_t = self._step_backward(gradOutput[:, :, i]+ da_prev_t, a_arr[:,:,i], a_arr[:,:,i-1], input[:, :, i])
            else:
                din_t, da_prev_t, dWa_t, dba_t = self._step_backward(gradOutput[:, :, i] + da_prev_t, a_arr[:, :, i],
                                                                     self._a0, input[:, :, i])
            din[:, :, i] = din_t
            self._dWa += dWa_t
            self._dba += dba_t
        self._dWa/=input.shape[1]
        self._dba/=input.shape[1]
        return din

    def update_params(self, learning_rate = 0.01):
        self._Wa -= learning_rate*self._dWa
        self._ba -= learning_rate*self._dba
        self._dWa[:, :] = 0
        self._dba[:, :] = 0
