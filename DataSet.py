"""
DataSet类提供对整个数据集的管理。
主要是划分数据集，提供batch。
"""
import copy
import logging

from Vocabulary import Vocabulary, SequenceVocabulary
import numpy as np


def padding(source_list, target_length, padding_token):
    target_list = copy.deepcopy(source_list)
    for element in target_list:
        length = len(element)
        while length < target_length:
            element.append(padding_token)
            length += 1
    return target_list


class SeqDataSet(object):
    def __init__(self, source_path, target_path, batch_size, vocabulary: Vocabulary = None):
        self._vocabulary = vocabulary if vocabulary is not None else SequenceVocabulary()
        self._source_path = source_path
        self._target_path = target_path
        self._batch_size = batch_size
        with open(source_path) as source:
            source_data = source.read().split('\n')

        tmp = [sentense.split(" ") for sentense in source_data]
        source_data = [[self._vocabulary.add_token(element) for element in sentense_split] for sentense_split in tmp]
        with open(target_path) as target:
            target_data = target.read().split('\n')
        tmp = [sentense.split(" ") for sentense in target_data]
        target_data = [[self._vocabulary.add_token(element) for element in sentense_split] for sentense_split in tmp]
        self.train_X = np.array(source_data[:int(len(source_data) * 4 / 5)])
        self.train_y = np.array(target_data[:int(len(source_data) * 4 / 5)])
        self.valid_X = np.array(source_data[int(len(source_data) * 4 / 5):int(len(source_data) * 9 / 10)])
        self.valid_y = np.array(target_data[int(len(source_data) * 4 / 5):int(len(source_data) * 9 / 10)])
        self.test_X = np.array(source_data[int(len(source_data) * 9 / 10):])
        self.test_y = np.array(target_data[int(len(source_data) * 9 / 10):])

    def train_batch_generator(self):
        logging.info("get_next_batch invoked")
        if self._batch_size > len(self.train_X):
            self._batch_size = len(self.train_X)

        shuffle_idx = np.random.permutation(range(len(self.train_X)))
        self.train_X = self.train_X[shuffle_idx]
        self.train_y = self.train_y[shuffle_idx]
        for base in range(0, len(self.train_X), self._batch_size):
            if base + self._batch_size > len(self.train_X):
                break
            batch_train_X = self.train_X[base:base + self._batch_size]
            batch_train_y = self.train_y[base:base + self._batch_size]
            batch_train_X_length = [len(sentence) for sentence in batch_train_X]
            batch_train_y_length = [len(sentence) for sentence in batch_train_y]
            batch_train_X = padding(batch_train_X.tolist(), 10, 0)
            batch_train_y = padding(batch_train_y.tolist(), 10, 0)
            yield (batch_train_X, batch_train_y, batch_train_X_length, batch_train_y_length)

    def valid_batch_generator(self):
        logging.info("valid_batch_generator")
        if self._batch_size > len(self.valid_X):
            self._batch_size = len(self.valid_y)

        shuffle_idx = np.random.permutation(range(len(self.valid_X)))
        self.valid_X = self.valid_X[shuffle_idx]
        self.valid_y = self.valid_y[shuffle_idx]
        for base in range(0, len(self.valid_X), self._batch_size):
            if base + self._batch_size > len(self.valid_X):
                break
            batch_valid_X = self.valid_X[base:base + self._batch_size]
            batch_valid_y = self.valid_y[base:base + self._batch_size]
            batch_valid_X_length = [len(sentence) for sentence in batch_valid_X]
            batch_valid_y_length = [len(sentence) for sentence in batch_valid_y]
            batch_valid_X = padding(batch_valid_X.tolist(), 10, 0)
            batch_valid_y = padding(batch_valid_y.tolist(), 10, 0)
            yield (batch_valid_X, batch_valid_y, batch_valid_X_length, batch_valid_y_length)


if __name__ == "__main__":
    vocabulary = SequenceVocabulary()
    seqDataSet = SeqDataSet('data/dataSource', 'data/dataTarget', 4, vocabulary)
    batch_generator = seqDataSet.train_batch_generator()
    (batch_train_X, batch_train_y, batch_train_X_length, batch_train_y_length) = next(batch_generator)
