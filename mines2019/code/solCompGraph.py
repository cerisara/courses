import torch

x = torch.tensor([1.],requires_grad=True)
y = torch.tensor([2.])
z = torch.tensor([3.])

loss = x*y*y*y + torch.pow(torch.abs(2+x*z),2)
loss.backward()
print(x.grad)

