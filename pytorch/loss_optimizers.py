import torch
import torch.nn as nn


# Losses are implemented as an nn.Module subclass. They take as input the output from the network (prediction) and the
# desired output (data) and return a real.
# Examples: nn.MSELoss, nn.BCEWithLogits, nn.CrossEntropyLoss, etc.

loss1 = nn.MSELoss()
loss2 = nn.MSELoss(reduction='sum')
v = torch.tensor([3.,6.])
u = torch.tensor([4.,2.])

print(loss1(u,v)) # 0.5*(3-4)**2 + 0.5*(6-2)**2 = 0.5 + 8 = 8.5
print(loss2(u,v)) # (3-4)**2 + (6-2)**2 = 1 + 16 = 17

# note on Variable (see torch.autograd): a Variable wraps a Tensor. It supports nearly all the API’s defined by a Tensor. Variable also provides
# a backward method to perform backpropagation.

# Optimizers: SGD, RMSprop, Adagrad, etc. (see torch.optim)

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
print(net1.parameters())

optimizer = torch.optim.SGD(net1.parameters(), lr = 0.1, momentum=0.9) # lr = learning rate