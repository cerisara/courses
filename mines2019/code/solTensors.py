import torch

# Use torch.randn to create two tensors of size a=(4, 3, 2) and b=(2).
a=torch.randn((4,3,2))
b=torch.randn((2,))

# Multiply each of the 4 (batch dimension) matrices of a with b
print(a)
print(b)
c=torch.matmul(a,b)
print(c.size())
print(c)
print(a[0,0,0].item()*b[0].item() + a[0,0,1].item()*b[1].item())
print(a[0,1,0].item()*b[0].item() + a[0,1,1].item()*b[1].item())

# find the argmax element (amongst the 3)
ci = torch.argmax(c,dim=1)
print(ci)


