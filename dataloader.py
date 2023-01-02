import io
import os
import string
import torch

import unicodedata

alphabet = string.ascii_letters + " .,;'"
n_letters = len(alphabet)


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn'
                   and c in alphabet)


def read_lines(filename):
    lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]


category_lines = {}
all_categories = []

for filename in os.listdir('data'):
    category = filename.split('.')[0]
    all_categories.append(category)
    lines = read_lines('data/' + filename)
    category_lines[category] = lines

n_categories = len(all_categories)


def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):  # One shot represent
        tensor[i][0][alphabet.find(letter)] = 1
    return tensor
