# Name-Region-Classifier

Name Region Classifier for dataset: https://download.pytorch.org/tutorial/data.zip
Based on Pytorch RNN, LSTM, GRU

Config with only one layer.

File Structure:

```
GRU.py RNN.py LSTM.py basic model of its name
SARNN.py use SA to optimize RNNs (GRU RNN)
train.py Main code to train and predict, also plot
dataloader.py pre-prossess the data
optimizer.py Use Adam and change LR rate to optimize
data the hole dataset
```

## Usage

```shell
$ python train.py # the 3 models
$ python SARNN.py # the improved model
```

**ATTENSION** the improved model is self-designed, some peremeters should be change to get the best preformance.
