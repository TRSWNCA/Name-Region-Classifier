import torch.nn as nn
import torch
import dataloader

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
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train(X, Y):
    hidden = torch.zeros(1, 1, 64)
    model.zero_grad()
    for i in range(X.size()[0]):
        out, hidden = model(X[i], hidden)

    loss = criterion(out.squeeze(0), Y)
    loss.backward()
    # for p in model.parameters():
    #     p.data.add_(-learning_rate, p.grad.data)
    opt.step()

    return out, loss.item()


def predict(X):
    hidden = torch.zeros(1, 1, 64)
    model.zero_grad()
    for i in range(X.size()[0]):
        out, hidden = model(X[i], hidden)

    return out

train_X, test_X, valid_X, train_Y, test_Y, valid_Y = dataloader.split_data()
for i in range(10):
    sum_loss = 0
    for j in range(len(train_X)):
        X = train_X[j]
        Y = train_Y[j]
        out, loss = train(X, Y)
        sum_loss += loss
    correct = 0
    for j in range(len(valid_X)):
        X = valid_X[j]
        Y = valid_Y[j]
        out = predict(X)
        guess, guess_i = dataloader.category_from_output(out)
        if guess_i == Y[0]:
            correct += 1
    print('epoch: %d, loss: %d, accuracy: %.2f' % (i + 1, sum_loss, correct / len(valid_X)))
