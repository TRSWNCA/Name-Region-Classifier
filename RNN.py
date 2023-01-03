import torch.nn as nn
import torch

hidden_size = 64
learning_rate = 0.01


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.model = nn.RNN(57, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, 18)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h):
        x = x.unsqueeze(0)
        rr, hn = self.model(x, h)
        return self.softmax(self.linear(rr)), hn


criterion = nn.NLLLoss()
model = RNN()


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
