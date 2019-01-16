import torch
import numpy as np


# N.B: PyTorch (unlike TensorFlow) uses a dynamic graph (as opposed to static), also called
# "notebook gradients", i.e., recalls all operations at execution and uses this to compute
# the final gradient (Tensorflow optimizes the fixed NN before executions)

t1 = torch.tensor([[1.,2.],[8.,0.],[5.,7.]], requires_grad=True) # requires_grad needs to be explicitly set to True if gradient is needed
t2 = torch.tensor([3.,2.])
t3 = torch.tensor([[0.,1.],[1.,1.],[0.,1.]])

print(t1.is_leaf) # tensor constructed by user is a "leaf" ("True" expected)

t_sum = t1 + t3 # first operation
print(t_sum.is_leaf) # tensor NOT constructed by user is NOT a "leaf" ("False" expected)

t_res = t_sum.sum()*3 + t2.sum()*2 # second operation

print(t_sum.requires_grad) # True because inherit from t1
print(t_res.requires_grad)

t_res.backward() # automatically computes the gradient of t_res with respect to all other variables
print(t1.grad)
print(t_sum.grad)

print(t2.grad) # return None, we need to set requires_grad to True at the begin

t_res2 = t_sum.sum()
t_res2.backward()

print(t1.grad) # the gradient will take into account all "backwards" calls (so in this code: both t_res and t_res2), the gradient is wrt (t_res + t_res2)