import torch
import numpy as np
import torch.nn as nn


# All modules of the torch.nn package follow the convention of callable, meaning that the instance of any class can
# act as function when applied to its arguments (see example below)

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

u = torch.FloatTensor([[1,8,2,6],[7,2,0,5]])
d = model(u)
print(d) # Try changing dim in Softmax and observe the result

# Custom layers

class customModule(nn.Module):
    def __init__(self, nb_inputs, nb_classes, dropout_prob=0.1):
        '''
        Register all submodules here i.e., all parameters
        Warning: If you want to use a list of nn, use nn.ModuleList() otherwise the parameters are not registered as such
        '''
        super(customModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(nb_inputs,3),
            nn.ReLU(),
            nn.Linear(3,5),
            nn.ReLU(),
            nn.Linear(5, nb_classes),
            nn.Dropout(p=dropout_prob),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.net(x)

net1 = customModule(nb_inputs=3, nb_classes=2)
v = torch.FloatTensor([4,7,2])
v.requires_grad = True
res = net1(v)
print(res)
print(net1)

res2 = res.sum()
print(res2)
res2.backward(retain_graph = True) # retain_graph=True is necessary when we want to compute a second gradient involving the same modules later (see res3 below)
print(v.grad) # Expected result = tensor([0, 0]) since res2 is always equal to 1 due to the Softmax layer!

res3 = res[1]
print(res3)
res3.backward()
print(v.grad)

