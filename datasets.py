import os
import numpy as np

import torch
from torch.utils.data import Dataset


class MNIST_Seq(Dataset):
    r"""Generates MNIST sequences of up to a given length.

    Reproduces the original approach by Zaheer et al. (Deep Sets):
    https://github.com/manzilzaheer/DeepSets/blob/master/DigitSum/image_sum.ipynb

    In this version, random sequences are created offline, i.e. all sequences are
    created when the dataset is instantiated and are repeated throughout the training.
    Images containing the digit 0 are skipped."""

    def __init__(self, pack=0, path='./infimnist_data', num_examples=150000,
                 max_seq_len=10, rand_seq_len=True):
        self.num_examples = num_examples
        self.rand_seq_len = rand_seq_len

        with open(os.path.join(path, 'mnist8m_{0}_features.bin'.format(pack)), 'r') as f:
            _ = np.fromfile(f, dtype='int32', count=1)
            D = np.asscalar(np.fromfile(f, dtype='int32', count=1))
            N = np.asscalar(np.fromfile(f, dtype='int32', count=1))
            arr = np.fromfile(f, dtype='int32').astype('float32')
        self.img_size = int(np.sqrt(D))
        self.img = np.reshape(arr, (N, 1, self.img_size, self.img_size))/255.

        with open(os.path.join(path, 'mnist8m_{0}_labels.bin'.format(pack)), 'r') as f:
            _ = np.fromfile(f, dtype='int32', count=1)
            D = np.asscalar(np.fromfile(f, dtype='int32', count=1))
            N = np.asscalar(np.fromfile(f, dtype='int32', count=1))
            arr = np.fromfile(f, dtype='int32').astype(int)
        self.label = np.reshape(arr, (N, D))

        rng_state = np.random.get_state()
        np.random.shuffle(self.img)
        np.random.set_state(rng_state)
        np.random.shuffle(self.label)

        self.set_max_seq_len(max_seq_len)

    def set_max_seq_len(self, max_seq_len):
        self.max_seq_len = max_seq_len
        self.index_matrix = -1*np.ones((self.num_examples, self.max_seq_len), dtype=int)
        self.lengths = np.zeros(self.num_examples, dtype=int)

        m = 0
        for i in range(self.num_examples):
            if self.rand_seq_len:
                self.lengths[i] = np.random.randint(1, self.max_seq_len+1)
            else:
                self.lengths[i] = self.max_seq_len
            for j in range(self.lengths[i]):
                while self.label[m] == 0:
                    m += 1
                self.index_matrix[i, j] = m
                m += 1

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        seq_len = self.lengths[index]
        X = torch.zeros(self.max_seq_len, 1, self.img_size, self.img_size)
        Y = torch.zeros(self.max_seq_len).long()

        for i in range(seq_len):
            j = self.index_matrix[index, i]
            X[i] = torch.from_numpy(self.img[j])
            Y[i] = torch.from_numpy(self.label[j]).long().unsqueeze(1)

        return X, Y, seq_len


class MNIST_SeqOnline(Dataset):
    r"""Generates MNIST sequences online of up to a given length.

    In this version, sequences are created online, i.e. a new random sequence is
    created whenever __getitem__ is called. Images containing the digit 0 are NOT
    skipped."""

    def __init__(self, pack=0, path='./infimnist_data/', num_examples=None,
                 max_seq_len=10, rand_seq_len=True):
        self.rand_seq_len = rand_seq_len

        with open(os.path.join(path, 'mnist8m_{0}_features.bin'.format(pack)), 'r') as f:
            _ = np.fromfile(f, dtype='int32', count=1)
            D = np.asscalar(np.fromfile(f, dtype='int32', count=1))
            N = np.asscalar(np.fromfile(f, dtype='int32', count=1))
            arr = np.fromfile(f, dtype='int32').astype('float32')
        self.img_size = int(np.sqrt(D))
        self.img = np.reshape(arr, (N, 1, self.img_size, self.img_size))/255.

        with open(os.path.join(path, 'mnist8m_{0}_labels.bin'.format(pack)), 'r') as f:
            _ = np.fromfile(f, dtype='int32', count=1)
            D = np.asscalar(np.fromfile(f, dtype='int32', count=1))
            N = np.asscalar(np.fromfile(f, dtype='int32', count=1))
            arr = np.fromfile(f, dtype='int32').astype(int)
        self.label = np.reshape(arr, (N, D))

        rng_state = np.random.get_state()
        np.random.shuffle(self.img)
        np.random.set_state(rng_state)
        np.random.shuffle(self.label)

        if num_examples:
            self.num_examples = num_examples
            self.img = self.img[:self.num_examples]
            self.label = self.label[:self.num_examples]
        else:
            self.num_examples = N

        self.set_max_seq_len(max_seq_len)

    def set_max_seq_len(self, max_seq_len):
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        if self.rand_seq_len:
            seq_len = np.random.randint(1, self.max_seq_len + 1)
        else:
            seq_len = self.max_seq_len
        X = torch.zeros(self.max_seq_len, 1, self.img_size, self.img_size)
        Y = torch.zeros(self.max_seq_len).long()
        for i in range(seq_len):
            j = (index
                 + np.random.randint(1, self.num_examples)) % self.num_examples
            X[i] = torch.from_numpy(self.img[j])
            Y[i] = torch.from_numpy(self.label[j]).long().unsqueeze(1)

        return X, Y, seq_len
