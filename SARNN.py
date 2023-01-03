import math
import random
import matplotlib.pyplot as plt
import RNN

import torch.nn as nn
import torch

import dataloader

hidden_size = 64

learning_rate = 0.01
torch.manual_seed(22)


class SARNN(nn.Module):
    def __init__(self):
        super(SARNN, self).__init__()
        self.model = nn.RNN(57, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, 18)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h):
        x = x.unsqueeze(0)
        rr, hn = self.model(x, h)
        return self.softmax(self.linear(rr)), hn


# train_X, test_X, valid_X, train_Y, test_Y, valid_Y = dataloader.split_data()
criterion = nn.NLLLoss()

EPOCH = 30

# train_X = train_X[:1000]
# train_Y = train_Y[:1000]


model = SARNN()


def train(X, Y):
    hidden = torch.zeros(1, 1, 64)
    model.zero_grad()
    for i in range(X.size()[0]):
        out, hidden = model(X[i], hidden)

    loss = criterion(out.squeeze(0), Y)
    loss.backward()
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return out, loss.item()


def predict(X):
    hidden = torch.zeros(1, 1, 64)
    model.zero_grad()
    for i in range(X.size()[0]):
        out, hidden = model(X[i], hidden)

    guess, guess_i = dataloader.category_from_output(out)
    return guess_i


gru_val = []
gru_loss = []


def all(train_X, train_Y, valid_X, valid_Y):
    global model, learning_rate
    sum_loss = 0
    for i in range(EPOCH):
        learning_rate, old = learning_rate + (random.randint(1, 100) - 70) / 2000 / 10000 * sum_loss, learning_rate
        sum_loss = 0
        torch.save(model, 'model.pkl')
        for j in range(len(train_X)):
            X = train_X[j]
            Y = train_Y[j]
            out, loss = train(X, Y)
            sum_loss += loss
        correct = 0
        if len(gru_loss) > 0:
            if gru_loss[-1] < sum_loss:
                val = math.exp(-(sum_loss - gru_loss[-1]) / (EPOCH - i) / 500)
                print(val)
                if random.randint(0, 99) > val * 80:
                    model = torch.load('model.pkl')
                    learning_rate = old
                    print('Decline %.2f %.2f' % (sum_loss, gru_loss[-1]))
                    sum_loss = gru_loss[-1]
        for j in range(len(valid_X)):
            X = valid_X[j]
            Y = valid_Y[j]
            guess_i = predict(X)
            if guess_i == Y[0]:
                correct += 1
        gru_val.append(correct / len(valid_X))
        gru_loss.append(sum_loss)
        print(
            'epoch: %d, loss: %d, accuracy: %.2f, lr: %.5f' % (i + 1, sum_loss, correct / len(valid_X), learning_rate))
    return gru_loss, gru_val

# rnn_val = []
# rnn_loss = []
#
# for i in range(EPOCH):
#     sum_loss = 0
#     for j in range(len(train_X)):
#         X = train_X[j]
#         Y = train_Y[j]
#         out, loss = RNN.train(X, Y)
#         sum_loss += loss
#     correct = 0
#     for j in range(len(valid_X)):
#         X = valid_X[j]
#         Y = valid_Y[j]
#         out = RNN.predict(X)
#         guess, guess_i = dataloader.category_from_output(out)
#         if guess_i == Y[0]:
#             correct += 1
#     rnn_val.append(correct / len(valid_X))
#     rnn_loss.append(sum_loss)
#     print('epoch: %d, loss: %d, accuracy: %.2f' % (i + 1, sum_loss, correct / len(valid_X)))
# x = [i + 1 for i in range(EPOCH)]
#
# plt.figure(0)
# # plt.plot(x, gru_val, label='RNN-SA')
# plt.plot(x, rnn_val, label='RNN')
# plt.title('Accuracy')
# plt.legend(loc='upper left')
# plt.savefig('Compare Accuracy.png')
#
# plt.figure(1)
# # plt.plot(x, gru_loss, label='RNN-SA')
# plt.plot(x, rnn_loss, label='RNN')
# plt.title('Loss')
# plt.legend(loc='upper left')
# plt.savefig('Compare Loss.png')
