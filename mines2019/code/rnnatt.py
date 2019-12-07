import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

# define a RNN model that takes as input a curve, and predicts the next point
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hiddensize =20
        self.n_layers = 1
        self.rnn = nn.RNN(1, self.hiddensize, self.n_layers, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(self.hiddensize, 2)

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
        _, hidden = self.rnn(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(hidden.view(-1,self.hiddensize))

        return out

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hiddensize)
        return hidden

class ModelAtt(Model):
    def __init__(self):
        super(ModelAtt, self).__init__()
        qnp = 0.1*np.random.rand(self.hiddensize)
        self.q = nn.Parameter(torch.Tensor(qnp))

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        steps, last = self.rnn(x, hidden)
        alpha = torch.matmul(steps,self.q)
        alpha = nn.functional.softmax(alpha,dim=1)
        alpha2 = alpha.unsqueeze(-1).expand_as(steps)
        weighted = torch.mul(steps, alpha2)
        rep = weighted.sum(dim=1)
        out = self.fc(rep)
        return out, alpha

# create training corpus: a linear fonction
def f(x,offset):
    return 0.3*math.sin(0.1*x+offset)+0.5

nex=100
nsteps=50
input_seqs = []
target_seqs = []
for ex in range(nex):
    offset = np.random.rand()
    input_seq=[f(x,offset) for x in range(nsteps)]
    cl = np.random.randint(2)
    target_seqs.append(cl)
    if cl==0: perturb = 0.05
    else: perturb = -0.05
    pos=np.random.randint(25,45)
    for t in range(pos,pos+5): input_seq[t]+=perturb
    input_seqs.append(input_seq)

# Convert all this into pytorch tensors
input_seq = torch.Tensor(input_seqs)
input_seq = input_seq.view(nex,nsteps,1)
target_seq = torch.LongTensor(target_seqs)

# Instantiate the model
model = ModelAtt()

n_epochs = 10000
lr=0.0001

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

plt.plot(input_seq[0].view(-1).numpy())
plt.show()

# Training Run
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    output,_ = model(input_seq)
    loss = criterion(output, target_seq)
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly

    if epoch%1 == 0:
        print('Epoch: %d/%d............. Loss %f' % (epoch, n_epochs, loss.item()))
        if loss.item()<0.2: break

_, alpha = model(input_seq)
for i in range(5):
    a = alpha[i].view(-1).detach().numpy()
    amax = max(a)
    amin = min(a)
    a = (a-amin)*0.5/(amax-amin)
    y = input_seq[i].view(-1).numpy()

    plt.plot(y)
    plt.plot(a)
    plt.show()

