import torch as t
import os
from dataset import DataSet
from net.DeepUNet import DeepUNet
from net.DeepUNetFIne import DeepUNetV2
from net.DenseUNet import FCDenseNet103,FCDenseNet57
from config import DefaultConfig
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from loss import CrossEntropyLoss2d 


opt = DefaultConfig()


def test_data(net):
    
    model =  net
    
   # model.eval()
    test_data = DataSet('data/', train=False, test=True)
    test_DataLoader =  DataLoader(test_data, 1, shuffle=False)
    index = 0
    for data,label in test_DataLoader:
        data = Variable(data)
        data = data.cuda()
        target = model(data)
       # loss = criterion(target, label)
       # loss_all = loss_all + loss.data[0]
       # loss.backward()
        target = F.log_softmax(target)
        target = (target.data).cpu()
        cal_f1(target, label)
        index = index + 1
           

def cal_f1(target, label):
    a = target[0].numpy()
    b = target[1].numpy()
    c = target[2].numpy()
    d = np.maximum(a,b)
    index = np.maximum(d,c)
    result = index.copy()
    result[index==a] = 0
    result[index==b] = 1
    result[index==c] = 1
    result = result.reshape(-1)
    
    label = label.numpy()
    label = label.reshape(-1)
    for view in range(0,255):
        label[label<view] = 0
    f1 = metrics.f1_score(label, result)
    return np.uint8(result)
        
#for i in range(142,178):
test_data('0.pkl')
        
