from sklearn.base import BaseEstimator
from collections import defaultdict
import nltk
import numpy as np

nltk.download('punkt')


class CustomPreProcess(BaseEstimator):

    def default_val(self):  # Pickle can't serialize lambda
        return 0

    def __init__(self, keep_nbr=200, debug=False):
        self.keep_nbr = keep_nbr
        self.word_to_idx = defaultdict(self.default_val)
        self.num_cols = 0  # searching for non existing index adds id to word_to_idx => len increases after transform
        self.debug = debug
        pass

    def tokenizing(self, X):
        return [nltk.word_tokenize(sentence.lower()) for sentence in X]

    def make_dictionary(self, rows):
        # Getting word frequency
        word_freq = {}
        for sentence in rows:
            for word in sentence:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1

        # Reducing dictionary size based on word frequency
        word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))
        words = [word for word, freq in word_freq.items() if freq > self.keep_nbr]

        # Creating dictionary
        idx = 1
        for word in words:
            self.word_to_idx[word] = idx
            idx += 1

        self.num_cols = len(self.word_to_idx) + 1

    def fit(self, X, y=None):
        if self.debug:
            print('Tokenizing... Might take some time')
        split_X = self.tokenizing(X)
        if self.debug:
            print('Making dictionary')
        self.make_dictionary(split_X)
        if self.debug:
            print('Done!')
        pass

    def transform(self, X):
        if self.debug:
            print('Tokenizing... Might take some time')
        split_X = self.tokenizing(X)

        if self.debug:
            print('Transforming to vec')
        num_samples = X.shape[0]

        features_vec = np.zeros((num_samples, self.num_cols), dtype=np.int8)
        for sample_idx in range(num_samples):
            for word in split_X[sample_idx]:
                word_idx = self.word_to_idx[word]
                features_vec[sample_idx][word_idx] = 1

        return features_vec

    def fit_transform(self, X, y=None):
        self.fit(X, y),
        return self.transform(X)