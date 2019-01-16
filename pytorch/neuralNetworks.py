import torch
import numpy as np
import torch.nn as nn


# All modules of the torch.nn package follow the convention of callable, meaning that that the instance of any class can
# act as function when appliet to its arguments (see example below)

l = nn.Linear(4,3) # Warning! First arg is input size, second is output size (feed forward). Transformation is y = x * A + b  (y, b = row vector)
v = torch.FloatTensor([3, 1, 6, 2])
a = l(v)
print(a)

b = a.sum()
b.backward()

# N.B: All classes in torch.nn inherit from torch.nn.Modules (which should be used to create own NN block), see doc for functions implemented by all children

# Very important and convenient class: Sequential (allows building a NN very quickly)

model = nn.Sequential(nn.Linear(4,3),
                      nn.ReLU(),
                      nn.Linear(3,5),
                      nn.ReLU(),
                      nn.Linear(5,2),
                      nn.Dropout(p=0.3),
                      nn.Softmax(dim=0))

c = model(v)
print(c)

# Custom layers



