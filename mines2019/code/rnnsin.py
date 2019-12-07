import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

# define a RNN model that takes as input a curve, and predicts the next point
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hiddensize = 100
        self.n_layers = 1
        self.rnn = nn.RNN(1, self.hiddensize, self.n_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(self.hiddensize, 1)

        # special init: predict inputs
        # self.rnn.load_state_dict({'weight_ih_l0':torch.Tensor([[1.]]),'weight_hh_l0':torch.Tensor([[0.]]),'bias_ih_l0':torch.Tensor([0.]),'bias_hh_l0':torch.Tensor([0.])},strict=False)
        print(self.rnn.state_dict())
        # self.fc.load_state_dict({'weight':torch.Tensor([[1.]]),'bias':torch.Tensor([0.])},strict=False)
        print(self.fc.state_dict())

    def forward(self, x):

        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hiddensize)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hiddensize)
        return hidden

# create training corpus: a linear fonction
def f(x):
    return 0.3*math.sin(0.1*x)+0.5
nex=100
input_seq=[f(x) for x in range(nex)]
target_seq=[f(x+10) for x in range(nex)]

# Convert all this into pytorch tensors
input_seq = torch.Tensor(input_seq)
input_seq = input_seq.view(1,nex,1)
target_seq = torch.Tensor(target_seq).view(1,nex,1)

# Instantiate the model
model = Model()

n_epochs = 1000
lr=0.001

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

plt.plot(target_seq.view(-1).numpy())
plt.plot(input_seq.view(-1).numpy())
plt.show()

# Training Run
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    output, hidden = model(input_seq)
    loss = criterion(output, target_seq)
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly

    if epoch%1 == 0:
        print('Epoch: %d/%d............. Loss %f' % (epoch, n_epochs, loss.item()))
plt.plot(target_seq.view(-1).numpy())
plt.plot(output.view(-1).detach().numpy())
plt.show()


