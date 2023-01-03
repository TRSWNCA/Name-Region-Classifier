import io
import os
import random
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


def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, letter in enumerate(line):  # One hot represent
        tensor[i][0][alphabet.find(letter)] = 1
    return tensor


category_lines = {}
all_categories = []
all_x = []
all_y = []

def category_from_output(output):
    topn, topi = output.topk(1)
    category_i = topi[0].item()
    return all_categories[category_i], category_i


for filename in os.listdir('data'):
    category = filename.split('.')[0]
    all_categories.append(category)
    lines = read_lines('data/' + filename)
    category_lines[category] = lines
    for line in lines:
        all_x.append(line_to_tensor(line))
        all_y.append(torch.tensor([all_categories.index(category)], dtype=torch.long))


def random_train_example():
    category = random.choice(all_categories)
    line = random.choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)
    return line_tensor, category_tensor

def split_data():
    c = list(zip(all_x, all_y))
    random.seed(22)
    random.shuffle(c)
    all_x[:], all_y[:] = zip(*c)
    return all_x[:16000], all_x[16000:19000], all_x[19000:],\
           all_y[:16000], all_y[16000:19000], all_y[19000:]
