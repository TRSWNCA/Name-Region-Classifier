import torch.nn as nn
import torch

hidden_size = 64
learning_rate = 0.01


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.model = nn.LSTM(57, hidden_size, 1)
        self.linear = nn.Linear(hidden_size, 18)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h, c):
        x = x.unsqueeze(0)
        rr, (hn, cn) = self.model(x, (h, c))
        return self.softmax(self.linear(rr)), hn, cn


model = LSTM()
criterion = nn.NLLLoss()


def train(X, Y):
    c = hidden = torch.zeros(1, 1, 64)
    model.zero_grad()
    for i in range(X.size()[0]):
        out, hidden, c = model(X[i], hidden, c)

    loss = criterion(out.squeeze(0), Y)
    loss.backward()
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return out, loss.item()


def predict(X):
    c = hidden = torch.zeros(1, 1, 64)
    model.zero_grad()
    for i in range(X.size()[0]):
        out, hidden, c = model(X[i], hidden, c)

    return out
