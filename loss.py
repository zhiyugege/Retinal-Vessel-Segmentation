import torch.nn.functional as F
from torch import nn
import torch 
import numpy
from torch.autograd import Variable


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
#    
#_inputs = [[[[1,2,3],[2,3,4]],[[0.1,2,3],[2,6,4]]]]
#label = [[[0,0,1],[0,0,0]]]
#_inputs = torch.from_numpy(numpy.array(_inputs).astype(numpy.float))
#label = torch.from_numpy(numpy.array(label))
#loss = CrossEntropyLoss2d()
#a = Variable(_inputs)
#b = Variable(label)
#print(_inputs,label)
#print(F.softmax(a))
#print(loss(a,b).data[0])