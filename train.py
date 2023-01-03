import time
import dataloader
import RNN
import GRU
import LSTM
import matplotlib.pyplot as plt

train_X, test_X, valid_X, train_Y, test_Y, valid_Y = dataloader.split_data()

EPOCH = 30

lstm_loss = []
gru_loss = []
rnn_loss = []
lstm_val = []
gru_val = []
rnn_val = []

print("LSTM Begin")
st = time.time()
for i in range(EPOCH):
    sum_loss = 0
    for j in range(len(train_X)):
        X = train_X[j]
        Y = train_Y[j]
        out, loss = LSTM.train(X, Y)
        sum_loss += loss
    correct = 0
    for j in range(len(valid_X)):
        X = valid_X[j]
        Y = valid_Y[j]
        out = LSTM.predict(X)
        guess, guess_i = dataloader.category_from_output(out)
        if guess_i == Y[0]:
            correct += 1
    lstm_val.append(correct / len(valid_X))
    lstm_loss.append(sum_loss)
    print('epoch: %d, loss: %d, accuracy: %.2f' % (i + 1, sum_loss, correct / len(valid_X)))
ed = time.time()
t3 = ed-st
print(ed - st)
print("LSTM End")

print("GRU Begin")
st = time.time()
for i in range(EPOCH):
    sum_loss = 0
    for j in range(len(train_X)):
        X = train_X[j]
        Y = train_Y[j]
        out, loss = GRU.train(X, Y)
        sum_loss += loss
    correct = 0
    for j in range(len(valid_X)):
        X = valid_X[j]
        Y = valid_Y[j]
        out = GRU.predict(X)
        guess, guess_i = dataloader.category_from_output(out)
        if guess_i == Y[0]:
            correct += 1
    gru_val.append(correct / len(valid_X))
    gru_loss.append(sum_loss)
    print('epoch: %d, loss: %d, accuracy: %.2f' % (i + 1, sum_loss, correct / len(valid_X)))
ed = time.time()
t2 = ed-st
print(ed - st)
print("GRU End")

print("RNN Begin")
st = time.time()
for i in range(EPOCH):
    sum_loss = 0
    for j in range(len(train_X)):
        X = train_X[j]
        Y = train_Y[j]
        out, loss = RNN.train(X, Y)
        sum_loss += loss
    correct = 0
    for j in range(len(valid_X)):
        X = valid_X[j]
        Y = valid_Y[j]
        out = RNN.predict(X)
        guess, guess_i = dataloader.category_from_output(out)
        if guess_i == Y[0]:
            correct += 1
    rnn_val.append(correct / len(valid_X))
    rnn_loss.append(sum_loss)
    print('epoch: %d, loss: %d, accuracy: %.2f' % (i + 1, sum_loss, correct / len(valid_X)))
ed = time.time()
t1 = ed-st
print(ed - st)
print("RNN End")


x = [i + 1 for i in range(EPOCH)]
plt.figure(0)
plt.plot(x, gru_val, label='GRU', color="purple")
plt.plot(x, lstm_val, label='LSTM', color="blue")
plt.plot(x, rnn_val, label='RNN', color="green")
plt.title('Accuracy')
plt.legend(loc='upper left')
plt.savefig('Accuracy.png')

plt.figure(1)
plt.plot(x, gru_loss, label='GRU', color="purple")
plt.plot(x, lstm_loss, label='LSTM', color="blue")
plt.plot(x, rnn_loss, label='RNN', color="green")
plt.title('Loss')
plt.legend(loc='upper left')
plt.savefig('Loss.png')

# plt.figure(2)
# plt.plot(x, rnn_loss, label='RNN', color="green")
# plt.plot(x, sa_loss, label='RNN-SA', color="red")
# plt.title('Loss')
# plt.legend(loc='upper left')
# plt.savefig('Compare Loss.png')
#
# plt.figure(3)
# plt.plot(x, rnn_val, label='RNN', color="green")
# plt.plot(x, sa_val, label='RNN-SA', color="red")
# plt.title('Accuracy')
# plt.legend(loc='upper left')
# plt.savefig('Compare Accuracy.png')
#
plt.figure(4)
x = ['RNN', 'LSTM', 'GRU']
y = [t1, t3, t2]
plt.bar(range(len(x)), y, tick_label=x)
plt.title('Time cost')
plt.savefig('Time.png')



