import numpy as np
from easydict import EasyDict as Dict

import Dataset as dataset


class Network:
    def __str__(self):
        return "<object> RNN neural network for language model"
    def __init__(self, params):
        self.input_size = params.dimensions[0]
        self.hidden_size = params.dimensions[1]
        self.output_size = params.dimensions[2]

        # self.w_frist = np.random.random((self.hidden_size, self.input_size))
        # self.w_reverse = np.random.random((self.hidden_size, self.hidden_size))
        # self.w_second = np.random.random((self.input_size, self.hidden_size))

        # Randomly initialize the network parameters
        self.w_frist = np.random.uniform(-np.sqrt(1. / self.input_size), np.sqrt(1. / self.input_size), (self.hidden_size, self.input_size))
        self.w_second = np.random.uniform(-np.sqrt(1. / self.hidden_size), np.sqrt(1. / self.hidden_size), (self.input_size, self.hidden_size))
        self.w_reverse = np.random.uniform(-np.sqrt(1. / self.hidden_size), np.sqrt(1. / self.hidden_size), (self.hidden_size, self.hidden_size))

    def forward(self, x):
        # The total number of time steps
        T = len(x)
        # print("the length of sentence(number of time step) is, ", T)
        # During forward propagation we save all hidden states in s because need them later.
        # we add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_size))
        s[-1] = np.zeros(self.hidden_size)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.input_size))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.w_frist[: ,x[t]] + self.w_reverse.dot(s[ t -1]))
            o[t] = self.softmax(self.w_second.dot(s[t]))

        return [s, o]

    def softmax(self, vector):
        exp_vector = np.exp(vector)
        return exp_vector / np.sum(exp_vector)

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        s, o = self.forward(x)
        return np.argmax(o, axis=1)

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x ,y ) /N


    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            s, o = self.forward(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L


# defining the dimension for input_size, hidden_size and output_size
# we have 10000 dictionary of word
vocabulary_size = 10000
hidden_size = 100
params = Dict({
    "dimensions": [vocabulary_size, hidden_size, vocabulary_size]
})


# initializing the Network.py
network = Network(params)

print("\n\n\t-----------SHAPE OF THE WEIGHTS-------")
print('\n----------------------')
print("shape of w_first  ", network.w_frist.shape)
print('shape of w_second  ', network.w_second.shape)
print('shape of w_reverse ', network.w_reverse.shape)
print('---------------------------\n')


print("\n\n\t-------------TRAINING THE MODEL-----------\n")
# training the model
j = 1
for i in dataset.X_train[:2]:
    print("Sentence number ", j)
    print("input sentence \n", i)
    print("length of input, ", len(i))
    p = network.predict(i)
    print("shape of predictions ", p.shape)
    print("predictions,\n", p)
    print("-----------------------\n")
    j = j + 1


print("\n\n\t-------------THE LOSS--------------\n")
# lets limit to 100 examples to save time
print("Expected Loss for random predictions: %f" % np.log(vocabulary_size))
print("please wait about 2 minute for the Actual Loss to be displayed.....")
print("Actual loss: %f" % network.calculate_loss(dataset.X_train[:100], dataset.Y_train[:100]))