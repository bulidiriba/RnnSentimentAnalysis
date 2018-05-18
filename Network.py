import numpy as np
from easydict import EasyDict as Dict


class Network:
    def __str__(self):
        return "<object> RNN neural network for language model"
    def __init__(self, params):
        self.input_size = params.dimensions[0]
        self.hidden_size = params.dimensions[1]
        self.output_size = params.dimensions[2]
        self.w_frist = np.random.random((self.hidden_size, self.input_size))
        self.w_reverse = np.random.random((self.hidden_size, self.hidden_size))
        self.w_second = np.random.random((self.input_size, self.hidden_size))

    def forward(self, x):
        # define hidden_state with its shape the shape of hidden_state is input_size + 1 by hidden_size
        state = np.zeros((self.input_size + 1, self.hidden_size))

        # s[-1] is added on the last index means on the input_size index so s[-1] = s[input_size]
        # and the dimension of s[-1] is (hidden_dim) because each
        # s[t] dimension is (hidden_dim)
        state[-1] = np.ones((self.hidden_size))

        # define output and its shape the shape of output is similar to input means input_size by input_size
        output = np.zeros((self.input_size, self.input_size))

        # now we calculate each hidden_state and output for each time step (from 0 to input_size - 1)
        for t in np.arange(self.input_size):
            input = np.zeros(self.input_size)
            input[x[t]] = 1
            state[t] = np.tanh(self.w_frist.dot(input) + self.w_reverse.dot(state[t-1]))
            output[t] = self.softmax(self.w_second.dot(state[t]))

        return state, output

    def softmax(self, vector):
        exp_vector = np.exp(vector)
        return exp_vector / np.sum(exp_vector)


# defining the dimension for input_size, hidden_size and output_size
params = Dict({
    "dimensions": [3, 2, 3]
})


# initializing the Network
network = Network(params)
print('\n----------------------')
print("shape of w_first  ", network.w_frist.shape)
print('shape of w_second  ', network.w_second.shape)
print('shape of w_reverse ', network.w_reverse.shape)
print('---------------------------')

state, output = network.forward([2, 0, 1])
print('\n-------------------------')
print('the hidden state of the Model is \n', state)
print('\nshape of hidden state ', state.shape)
print('\n-------------------------')
print('the output of the Model is \n', output)
print('\nshape of outuput', output.shape)


