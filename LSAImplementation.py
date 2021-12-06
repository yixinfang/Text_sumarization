import collections
import math
from operator import attrgetter

import numpy
import nltk
from collections import namedtuple
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from numpy.linalg import svd
from rouge import Rouge
from tqdm import tqdm

import BARTeval
from DataLoader import task1and2Loader, task1and2ReferenceLoader

# Tuple to store the sentence information
SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating",))

"""
Counter class used to sort lists later on
"""


class Counter(object):
    def __init__(self, value):
        self.value = value

    def __call__(self, sequence):
        if isinstance(self.value, (bytes, str,)):
            return sequence[:int(self.value)]
        elif isinstance(self.value, (int, float)):
            return sequence[:int(self.value)]
        else:
            ValueError("Format is incorrect")


class LSASummarizer(object):
    stop_words_list = list(stopwords.words('english'))

    # Returns the list of stop words used by the LSA summarizer
    @property
    def stop_words(self):
        return self.stop_words_list

    # Stores stop words
    @stop_words.setter
    def stop_words(self, words):
        self.stop_words_list = words

    """
    Given a document, and number of sentences to summarize documents.
    """

    def __call__(self, document, sentences_count):
        # Create dictionary of words and indexes in document
        dictionary = self.create_dictionary(document)
        # Use nltk sentence tokenizer
        sentences = sent_tokenize(document)

        # create matrix and store term frequency in LSA
        matrix = self.create_matrix(document, dictionary)
        matrix = self.calculate_term_frequency(matrix)
        # u, sigma, v = singular_value_decomposition(matrix, full_matrices=False)
        # Use SVD on matrix and compute sentence rankings and find best ranking
        u, sigma, v = svd(matrix)
        ranks = iter(self.compute_rankings(sigma, v))
        return self.get_best_sentences(sentences, sentences_count, lambda s: next(ranks))

    """
    Lower cases all words in a document to normalize it.
    """

    def normalize_word(self, word):
        return word.lower()

    """
    Find the best ranking sentences based on rankings using SVD.
    """

    def get_best_sentences(self, sentences, count, rating):
        ranking = (SentenceInfo(sentence, order, rating(sentence)) for order, sentence in enumerate(sentences))

        # sort sentences by rating in descending order
        ranking = sorted(ranking, key=attrgetter("rating"), reverse=True)
        count = Counter(count)
        ranking = count(ranking)
        # sort the best sentences by their order in document using the util class
        ranking = sorted(ranking, key=attrgetter("order"))
        return tuple(i.sentence for i in ranking)

    """
    Creates a dictionary to store unique words and their index values.
    """

    def create_dictionary(self, document):
        # Tokenize document and maps normalized words with the actual word
        words = word_tokenize(document)
        words = tuple(words)
        words = map(self.normalize_word, words)
        # use frozen set to place objects into dictionary where the key is the word and the index is the value
        unique_words = frozenset(w for w in words if w not in self.stop_words_list)
        d = dict((word, index) for index, word in enumerate(unique_words))
        return d

    """
    Creates the input matrix for LSA. Rows are number of occurrences of words in sentences and the column is the 
    sentence. Input is the document being summarized and a dictionary of word and row index values.
    """

    def create_matrix(self, document, dictionary):
        sentences = sent_tokenize(document)
        # initialize a matrix that is number of words by number of sentences long
        matrix = numpy.zeros((len(dictionary), len(sentences)))
        for column, sentence in enumerate(sentences):
            words = word_tokenize(sentence)
            # Keep track of how many times that word is used in a sentence.
            for word in words:
                if word.lower() in dictionary:
                    row = dictionary[word.lower()]
                    matrix[row, column] += 1
        return matrix

    """
    Calculates term frequency with the equation in the read me, it uses double normalization weighting scheme where the ratio of 
    a term count to max term count is multiplied by a smoothing variable. 
    """

    def calculate_term_frequency(self, matrix, alpha=0.5):
        # Bound smooth to be between 0 and 1
        if alpha < 0:
            alpha = 0
        if alpha > 1:
            alpha = 1

        # finds the max term frequency for each sentence
        max_term_frequencies = numpy.max(matrix, axis=0)
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                # set the maximum term frequency for the current column
                max_term_frequency = max_term_frequencies[col]
                # Only proceed for a non empty sentence
                if max_term_frequency != 0:
                    frequency = matrix[row, col] / max_term_frequency
                    matrix[row, col] = alpha + (1.0 - alpha) * frequency
        return matrix

    """
    Compute rankings of each sentence using SVD values. 
    """

    def compute_rankings(self, sigma, v_matrix):
        min_dimension = 3
        reduction_ratio = 1 / 1
        dimensions = max(min_dimension, len(sigma) * reduction_ratio)
        sigma_squared = tuple(s ** 2 if i < dimensions else 0.0 for i, s in enumerate(sigma))
        ranks = []
        # ranks the sentence based off sigma and V matrix values
        for column_vector in v_matrix.T:
            rank = sum(s * v ** 2 for s, v in zip(sigma_squared, column_vector))
            ranks.append(math.sqrt(rank))
        return ranks


def calculate_rouge(sum_dict, ref_dict, ref_name):
    rouge = Rouge()
    cur_ref = ref_dict[ref_name]
    summary_ordered_list = [i[1] for i in list(sorted(sum_dict.items(), key=lambda item: item[0]))]
    reference_ordered_list = [i[1] for i in list(sorted(cur_ref.items(), key=lambda item: item[0]))]
    rouge_dict = rouge.get_scores(summary_ordered_list, reference_ordered_list, avg=True)
    return rouge_dict


if __name__ == '__main__':
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    summarizer = LSASummarizer()
    stopword = list(stopwords.words('english'))
    summarizer.stop_words = stopword
    rouge = Rouge()

    dataset = task1and2Loader()
    ref = task1and2ReferenceLoader()

    # Task 1 summaries
    task1_dict = collections.defaultdict(dict)
    for task in tqdm(dataset):
        for doc in tqdm(dataset[task]):
            task1_dict[task][doc] = " ".join(summarizer(dataset[task][doc], 1))
    task1_summaries = {}
    for task in tqdm(task1_dict):
        task1_dict[task] = " ".join(list(task1_dict[task].values()))
        task1_summaries[task] = " ".join(summarizer(task1_dict[task], 6))

    # Task 2 summaries
    task2_summaries = {}
    for task in tqdm(dataset):
        doc_cluster = " ".join(list(dataset[task].values()))
        task2_summaries[task] = " ".join(summarizer(doc_cluster, 6))

    # Print out Rouge scores
    rouge_task1 = BARTeval.calculate_rouge(task1_summaries, ref, 'reference1')
    rouge_task2 = BARTeval.calculate_rouge(task2_summaries, ref, 'reference2')
    print("Task 1:", rouge_task1)
    print("Task 2:", rouge_task2)
