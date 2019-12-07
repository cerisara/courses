import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.l1 = nn.Linear(2,2)
        self.l2 = nn.Linear(2,1)

    def forward(self,x):
        y = self.l1(x)
        y = y.sigmoid()
        y = self.l2(y)
        y = y.sigmoid()
        return y

def exmlp():
    mod = MLP()
    loss = nn.MSELoss()
    opt = torch.optim.SGD(mod.parameters(),lr=1.)
    x=[[0,0],[0,1],[1,0],[1,1]]
    y=[0,1,1,0]
    for i in range(100000):
        j = np.random.randint(4)
        datax = torch.tensor([x[j]],dtype=torch.float32,requires_grad=False)
        datay = torch.tensor([y[j]],dtype=torch.float32,requires_grad=False)
        opt.zero_grad()
        yhat = mod(datax)
        lo = loss(yhat,datay)
        lo.backward()
        opt.step()
        print("%d %f" % (i,lo.item()))

def exand():
    w0,w1,b=0,0,0
    x=[[0,0],[0,1],[1,0],[1,1]]
    y=[0,0,0,1]

    for epo in range(10):
        print(epo)
        for i in range(4):
            haty = b+w0*x[i][0]+w1*x[i][1]
            if haty>0: haty=1
            else: haty=0
            print("w=%d %d %d x=%d %d haty=%d y=%d" % (w0,w1,b,x[i][0],x[i][1],haty,y[i]))
            if haty>y[i]:
                w0-=x[i][0]
                w1-=x[i][1]
                b-=1
                newy = b+w0*x[i][0]+w1*x[i][1]
            elif haty<y[i]:
                w0+=x[i][0]
                w1+=x[i][1]
                b+=1
                newy = b+w0*x[i][0]+w1*x[i][1]

def exxor():
    w0,w1,b=0,0,0
    x=[[0,0],[0,1],[1,0],[1,1]]
    y=[0,1,1,0]

    for epo in range(10):
        print(epo)
        for i in range(4):
            haty = b+w0*x[i][0]+w1*x[i][1]
            if haty>0: haty=1
            else: haty=0
            print("w=%d %d %d x=%d %d haty=%d y=%d" % (w0,w1,b,x[i][0],x[i][1],haty,y[i]))
            if haty>y[i]:
                w0-=x[i][0]
                w1-=x[i][1]
                b-=1
                newy = b+w0*x[i][0]+w1*x[i][1]
            elif haty<y[i]:
                w0+=x[i][0]
                w1+=x[i][1]
                b+=1
                newy = b+w0*x[i][0]+w1*x[i][1]

def notnorm():
    w0,w1,b=0,0,0
    x=[[0,0],[0,2],[2,0],[2,2]]
    y=[0,0,0,1]

    for epo in range(10):
        print(epo)
        for i in range(4):
            haty = b+w0*x[i][0]+w1*x[i][1]
            if haty>0: haty=1
            else: haty=0
            print("w=%d %d %d x=%d %d haty=%d y=%d" % (w0,w1,b,x[i][0],x[i][1],haty,y[i]))
            if haty>y[i]:
                w0-=x[i][0]
                w1-=x[i][1]
                b-=1
                newy = b+w0*x[i][0]+w1*x[i][1]
            elif haty<y[i]:
                w0+=x[i][0]
                w1+=x[i][1]
                b+=1
                newy = b+w0*x[i][0]+w1*x[i][1]


exmlp()

