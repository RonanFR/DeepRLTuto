import torch
import numpy as np


# We will just create tensors of various size and type
# N.B: in PyTorch, tensors with different types are represented by different classes!

a = torch.FloatTensor(3,2) # type = 32-bit float
a2 = torch.DoubleTensor(3,2) # type = 64-bit float
b = torch.ByteTensor(3,2) # type = 8-bit unsigned integer
c = torch.LongTensor(3,2) # type = 64-bit signed integer
print(a)
print(a2)

a.zero_() # clear the tensor
print(a)

# N.B:
# Inplace operations (ends with an underscore) => operate on the tensor itself, return that tensor
# Outplace operations => Leave the original tensor untouched, create a copy with the performed modifications and return
#                        that modified copy

d = torch.FloatTensor([[0,1],[2,3],[4,5]]) # create tensor using a python iterable
print(d)
print(d.sum())
print(torch.stack([d,a],0)) # concatenate sequence of tensors along new dimension
print(torch.cat([d,a],0)) # concatenate sequence of tensors in the given dimension
print(torch.transpose(d,0,1))

e = np.zeros((3,2)) # create tensor using numpy array
print(e)
f = torch.tensor(e)
print(f)

e = np.zeros((3,2), dtype = np.float32)
f = torch.tensor(e)
print(f)
print(f.sum())

s = torch.tensor(4) # scalar tensor (0-dimensional)
print(s)
print(s.item())

# Testing CUDA
# every tensor defined above is for CPU and has its GPU equivalent but we need to manually convert it
# we say that 'cpu' or 'cuda' (GPU) are 2 possible "devices" for tensors

ca = a.to('cuda') # use function to(device) where device is an instance of class torch.device
print(ca)
print(ca.device)
print(a.device)

cca = ca.to('cpu')
print(cca)
print(cca.device)