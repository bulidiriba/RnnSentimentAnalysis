import collections

from blinker._utilities import lazy_property

import numpy as np

class Dataset:
    def __str__(self):
        return "positive and negative sentimental dataset"

    def __init__(self, positive_dataset_path, negative_dataset_path, vocabulary_size=10000):
        self._positive_dataset_path = positive_dataset_path
        self._negative_dataset_path = negative_dataset_path
        self._vocabulary_size = vocabulary_size
        self.sentence_end = "END"
        self.sentence_begin = "BEGIN"
        self.rare_word = "RARE"
        self._sentences_list = []
        self._tokenized_sentence_list = []
        self._word_dictionary = {}
        self._vector_dataset = []

    @lazy_property
    def labeled_dataset(self):
        """
        reads positive and negative sentimental files.
        and constructs labels for each sentence 1for positive 0 for negative
        :first item in the labels is the sentiment of first item in dataset
        :return: tuple of dataset_list containing sentences and label_list containing sentiment
        """
        pos_data = []
        neg_data = []

        with open(self._positive_dataset_path, 'r', encoding="latin-1") as pos_file:
            for line in pos_file:
                pos_data.append(line)

        with open(self._negative_dataset_path, 'r', encoding="latin-1") as neg_file:
            for line in neg_file:
                neg_data.append(line)

        tot_data = pos_data + neg_data
        labels = [1] * len(pos_data) + [0] * len(neg_data)

        return tot_data, labels

    @lazy_property
    def word_dictionary(self):
        """
        assigns each word[redundant enough to be in the vocabulary] a numerical value
        builds a dictionary of words and their assigned numerical value
        :return: tuple of dictionary and word frequency
        """
        words = [w for line in self.labeled_dataset[0] for w in line.split()]

        count = [[self.rare_word, -1], [self.sentence_begin, -2], [self.sentence_end, -3]]
        count.extend(collections.Counter(words).most_common(self._vocabulary_size - 3))
        # I subtract -3 for the rare, begin and end was added because they increase vocabulary size by 3

        for entry in count:
            self._word_dictionary[entry[0]] = len(self._word_dictionary)
        return self._word_dictionary, count

    @lazy_property
    def vector_dataset(self):
        """
        builds vectorized data set and label
        :return: tuple of dataset and label
        """
        vectorized_data = []
        for sentence in self.labeled_dataset[0]:
            vectorized_data.append(self.sentence2vector(sentence))
        return vectorized_data, self.labeled_dataset[1]

    @lazy_property
    def language_model_dataset(self):
        """
        this property holds dataset for language models
        the label of each input is the copy of the input shifted to the the left
        sentence_start will be prepended to the input and sentence_end will be appended to the label
        :return:tuple of inputs and labels
        """
        # lets take the vectorized data from the vector_dataset
        vectorized_data = self.vector_dataset[0]
        input_data = []
        label_data = []

        for vector in vectorized_data:
            # for input add Sentence Start to the each vector and its index is on 1 postion
            input_vector = [1] + vector
            input_data.append(input_vector)

            # for label add Sentece End to each vector and its index is on 2 postion
            label_vector = vector + [2]
            label_data.append(label_vector)

        return input_data, label_data

    def word2index(self, word):
        """
        takes a word and returns the numerical value assigned to the word
        if the number doesn't exist in the word dictionary return 0[RARE WORD]
        :param word:
        :return index: assigned numerical value
        """
        if word in self.word_dictionary[0]:
            return self.word_dictionary[0][word]
        else:
            return self.word_dictionary[0][self.rare_word]

    def sentence2vector(self, sentence_list):
        """
        each word in the list will be converted to its numerical value
        :param sentence_list: a list of words[sentence]
        :return: a list of numbers
        """
        vector = []
        for word in sentence_list.split(" "):
            vector.append(self.word2index(word))

        return vector


dataset = Dataset("data/rt-polarity.pos", "data/rt-polarity.neg", 10000)
#
# for l,d in zip(dataset.labeled_dataset[1][:5],dataset.labeled_dataset[0][:5]):
#     print(l,d)
#
# print(dataset.word_dictionary[1])
#
# for l,d in zip(dataset.vector_dataset[1][:5],dataset.vector_dataset[0][:5]):
#     print(l,d)
#
# # display the language model
# for data, label in zip(dataset.language_model_dataset[0][:5], dataset.language_model_dataset[1][:5]):
#     print("input_sent\n  ", data)
#     print("expected_output\n  ", label)
#     print('--------------\n')


# changing the language model to the train_data
X = []
Y = []
for input_data, label_data in zip(dataset.language_model_dataset[0][:], dataset.language_model_dataset[1][:]):
    X.append(input_data)
    Y.append(label_data)
X_train = np.asarray(X)
Y_train = np.asarray(Y)

# print(X_train.shape)
# print(Y_train.shape)


