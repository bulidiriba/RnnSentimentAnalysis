import collections
import numpy as np
import copy
from blinker._utilities import lazy_property


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
        self._word_dictionary = []
        self._vector_dataset = []


    """
        The labeled_dataset functions
    """
    @lazy_property
    def labeled_dataset(self):
        """
        reads positive and negative sentimental files.
        and constructs labels for each sentence 1for positive 0 for negative
        :first item in the labels is the sentiment of first item in dataset
        :return: tuple of dataset_list containing sentences and label_list containing sentiment
        """
        # define the variable label_list that holds the label for negative and postive data
        label_list = []

        # since the UnicodeError will be raised use encoding = 'ISO-8859-1'
        with open(self._negative_dataset_path, encoding='ISO-8859-1') as f:
            l = f.readlines()
            self._sentences_list.append(l)
            label_list.append(0)
        with open(self._positive_dataset_path, encoding='ISO-8859-1') as f:
            l = f.readlines()
            self._sentences_list.append(l)
            label_list.append(1)

        return self._sentences_list, label_list

    """
        The Word dictionary functions
    """

    @lazy_property
    def word_dictionary(self):
        """
        assigns each word[redundant enough to be in the vocabulary] a numerical value
        builds a dictionary of words and their assigned numerical value
        :return: tuple of dictionary and word frequency
        """
        # in this when we read file from the data we donot separate them using readlines because
        # Our goal is just to find the frequency of each and every word in the data
        # so we read all of them together no need to separate or split as in labeled_dataset functions

        # lets declare the list called all_words
        all_sentences = []
        with open(self._negative_dataset_path, encoding='ISO-8859-1') as f:
            all_sentences.append(f.read())

        with open(self._positive_dataset_path, encoding='ISO-8859-1') as f:
            all_sentences.append(f.read())

        # to check the length of all_words
        # print(len(all_sentences))
        # to check the length of words in negative data
        # print(len(all_sentences[0])

        # since all_word becomes 2D list lets change to 1D means mix positive and negative data together
        # inorder to simplify counting the frequency of each word in both negative and positive data

        # so when you mix the two list the result becomes a String
        all_data = all_sentences[0] + all_sentences[1]

        # lets check type of all_data
        # print("\ntype of all_data = ", type(all_data))

        # lets check the length of all_data string
        # print("the length of all_data = ", len(all_data))

        # lets display the first 20 character in the all_data string
        # for character in all_data[:20]:
            # print(character)

        # now split the all_data string with space(" ") and add these split word to self._word_dictionary
        self._word_dictionary = all_data.split(" ")


        # lets check the type of self._word_dictionary
        # print("\ntype of self._word_dictionary = ", type(self._word_dictionary))

        # lets check the length of self._word_dictionary
        # print("length of self._word_dictionary = ", len(self._word_dictionary))
        # print("\n")

        # lets display the first 10 word dictionary
        # for word in self._word_dictionary[:10]:
        #    print(word)

        # display all word dictionary
        # print(self._word_dictionary)

        # now lets change the self._word_dictionary to the numpy array to find the frequency of each word using
        # the collections.Counter
        word_dict = np.array(self._word_dictionary)

        word_frequency = collections.Counter(np.squeeze((word_dict)))

        #change word_frequency Counter to the list
        word_frequency_list = word_frequency.most_common()

        # now declare the rare, begin, and end postion
        rare_dict = [[self.rare_word, -1], [self.sentence_begin, -2], [self.sentence_end, -3]]
        # declare word_f list that contains the combinations of rare_dict and word_frequency
        word_f = []
        word_f.append(rare_dict)
        word_f.append(word_frequency_list)
        return word_dict, word_f


    """
        The Vector_dataset functions
    """
    @lazy_property
    def vector_dataset(self):
        """
        builds vectorized data set and label
        :return: tuple of dataset and label
        """

        # first take word_frequncy form word_dictionary
        word_frequency = self.word_dictionary[1]

        # then order the list of word by their frequency
        word_list_by_their_frequency = []
        for entry in word_frequency[1]:
            word_list_by_their_frequency.append(entry[0])

        # now take the sentence list to Vectorize it
        sentences, label = self.labeled_dataset

        # to hold the vectorized data of all positive sentence
        vectorized_pos = []
        for positive_stmt in sentences[1]:
            # to hold the vectorized data of each positive sentence
            pos_lis_s = []
            for word in positive_stmt.split():
                if word in word_list_by_their_frequency:
                    pos_lis_s.append(3 + word_list_by_their_frequency.index(word))
                    # why we add 3 because there is also RARE, BEGIN and END
                else:
                    pos_lis_s.append(0)
                    # this means if that word is RARE add 0 as index
            # then append each vectorized sentence to one general list
            vectorized_pos.append(pos_lis_s)

        # to hold the vectorized data of all negative sentence
        vectorized_neg = []
        for neg_stmt in sentences[0]:
            # to hold the vectorized data of each negative sentence
            neg_lis_s = []
            for word in neg_stmt.split():
                if word in word_list_by_their_frequency:
                    neg_lis_s.append(3 + word_list_by_their_frequency.index(word))
                    # why we add 3 because there is also RARE, BEGIN and END
                else:
                    neg_lis_s.append(0)
                    # this means if that word is RARE add 0 as index

            vectorized_neg.append((neg_lis_s))

        # then vectorized data
        vectorized_data = []
        vectorized_data.append(vectorized_neg)
        vectorized_data.append(vectorized_pos)
        return vectorized_data


    @lazy_property
    def language_model_dataset(self):
        """
        this property holds dataset for language models
        the label of each input is the copy of the input shifted to the the left
        sentence_start will be prepended to the input and sentence_end will be appended to the label
        :return:tuple of inputs and labels
        """
        # lets take the vectorized data from the vector_dataset
        vectorized_data = self.vector_dataset
        input_data_pos = []
        label_data_pos = []

        input_data_neg = []
        label_data_neg = []

        for vector_neg in vectorized_data[0]:
            input_vector = [1] + vector_neg
            input_data_neg.append(input_vector)

            label_vector = vector_neg + [2]
            label_data_neg.append(label_vector)

        for vector_pos in vectorized_data[1]:
            input_vector = [1] + vector_pos
            input_data_pos.append(input_vector)

            label_vector = vector_pos + [2]
            label_data_pos.append(label_vector)

        # then input_data and label_data
        input_data = []
        input_data.append(input_data_neg)
        input_data.append(input_data_pos)

        label_data = []
        label_data.append(label_data_neg)
        label_data.append(label_data_pos)


        return input_data, label_data


    def word2index(self, word):
        """
        takes a word and returns the numerical value assigned to the word
        if the number doesn't exist in the word dictionary return 0[RARE WORD]
        :param word:
        :return index: assigned numerical value
        """
        pass

    def sentence2vector(self, sentence_list):
        """
        each word in the list will be converted to its numerical value
        :param sentence_list: a list of words[sentence]
        :return: a list of numbers
        """
        pass




#Initialize the Class Dataset to variable dataset
dataset = Dataset("data/rt-polarity.pos", "data/rt-polarity.neg", 10000)

# lets assign variable sentences_list to _sentences_list and label_list to label_list in labeled_dataset functions
sentences_list, label_list = dataset.labeled_dataset


# lets display the type and length of both statement list and label list
print('\n------Lets Display the type and length of statement list and label list------\n')
print("type of sentences_list = ", type(sentences_list))# but its 2D list
print("length of sentences_list = ", len(sentences_list)) # this has 2 list then the inner list has the negative and postive data simultaneously
print("length of negative sentences_list = ", len(sentences_list[0]))# this list contains the negative data
print("length of positive_sentences_list = ", len(sentences_list[1]))# this list contains the positive data
print("\n")
print("type of label_list = ", type(label_list)) # its list
print("length of label_list = ", len(label_list)) # it has only 2 value O and 1 only 0 corresponds to the negative data and 1 corresponds to the postive data
print("value of the label corresponds to negative_sentences_list = ", label_list[0]) # this is 0 which is the label for negative data
print("value of the label corresponds to poitive_sentences_list = ", label_list[1]) # this is 1 which is the label for positive data

# beer in mind that both a and b has the the same length which is 2

# lets display the first 5 postive data with their label
print("\n------Lets Display the first 5 postive data with its label-----\n")
for positive_stmt in sentences_list[1][:5]:
    print(label_list[1], " ", positive_stmt)

print("\n")

# lets display the first 5 negative data with their label
print("\n------Lets Display the first 5 negative data with its label-----\n")
for negative_stmt in sentences_list[0][:5]:
    print(label_list[0], " ", negative_stmt)

print("\n\n")

print("------------------------")
# lets assign the word_dictionary and frequency of the word to the below variable
word_dict, word_frequency = dataset.word_dictionary

# lets display the word_dictionary
# print("\n-----The all Words in Dictionary------\n\n", word_dict)

# lets display the frequency of the word
# print("\n----The Frequency of each word-------\n\n", word_frequency)

print('-----------------------------')

vectorized_data = dataset.vector_dataset

# note be patient because at least it will take around 5 minute to complete the vector_dataset function
for vector in vectorized_data[1][:5]:
    # add label 1 because we are displaying the vectorized_pos wich is vectorized_data[1]
    print('1  ', vector)

print('\n')
print('-----------------------------------')

input_data, label_data = dataset.language_model_dataset

# to display the firs five sentence in the postive sentence
for d, l in zip(input_data[1][:5], label_data[1][:5]):
    print(d)
    print(l)
    print('-------------------')