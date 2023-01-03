import torch.nn as nn
import torch

import dataloader

hidden_size = 64
learning_rate = 0.01


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.model = nn.GRU(57, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, 18)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h):
        x = x.unsqueeze(0)
        rr, hn = self.model(x, h)
        return self.softmax(self.linear(rr)), hn


criterion = nn.NLLLoss()

model = GRU()


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

    return out

# sum_loss = 0
#
# for i in range(10000):
#     X, Y = dataloader.random_train_example()
#     out, loss = train(X, Y)
#     sum_loss += loss
#     if i % 100 == 0:
#         guess, guess_i = dataloader.category_from_output(out)
#         print('epoch: %d, loss: %d, %d - %d' % (i / 100, sum_loss, guess_i, Y[0]))
#         sum_loss = 0
#

