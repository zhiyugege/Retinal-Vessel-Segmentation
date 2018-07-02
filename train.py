from net.DeepUNet import DeepUNet
from net.DeepUNetFIne import DeepUNetV2
from net.DenseUNet import FCDenseNet103
from dataset import DataSet
from torch.utils.data import DataLoader
from config import DefaultConfig
from torch.autograd import Variable
import torch as t
import numpy as np
import torch.nn.functional as F
from loss import CrossEntropyLoss2d
#from test import test_data
#import visdom


opt = DefaultConfig()
#vis = visdom.Visdom(env=u'retina')

position = []
loss_arr = []


def train():
    
    i=0
    net =  FCDenseNet103(1,3)
   # net =  DeepUNetV2()
   # net.load_state_dict(t.load('./models/142.pkl'))
    net = net.cuda()
    net.train()
    train_data = DataSet('data/', train=True)

#    loss_weight = t.from_numpy(np.array(opt.loss_weight))
#    loss_weight = (loss_weight.float()).cuda()
  
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    criterion = CrossEntropyLoss2d()
    
    optimizer = t.optim.SGD(net.parameters(), lr=opt.lr, weight_decay = opt.weight_decay)
    for epoch in range(opt.max_epoch):
        loss_all = 0
        num = 0
        for data,label in train_dataloader:

            inputs = Variable(data)
            target = Variable(label)
            target = target.long()
            if opt.use_GPU:
                
                inputs = inputs.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            pre_target = net(inputs)
            loss = criterion(pre_target, target)
            loss.backward()
            optimizer.step()
            loss_all = loss_all + loss.data[0]
            i = i+1
            num = num+1
#            visdomGUI(i,loss, F.log_softmax(pre_target), target)
        now = 143
        per_loss = float(loss_all / num)
        print("The {} epoch loss is:{}".format(epoch,per_loss))
        _dir = './models/'
        t.save(net.state_dict(),_dir+str(epoch+now)+'.pkl')
        t.save(net,_dir+str(epoch+now)+'.pth')


def visdomGUI(i, loss, pre_target, target):
    
    position.append(i)
    loss_arr.append(loss.data[0])
    vis.line(X=np.array(position),Y=np.array(loss_arr),win='train_loss')
    target_show = (pre_target.data).cpu()
    target_show = get_max(target_show[0])
    vis.image(target_show, win='train_target')
    target = (target.data).cpu()
    target = get_label(target[0])
    vis.image(target,win='label')

def get_max(target):
    a = target[0].numpy()
    b = target[1].numpy()
    c = target[2].numpy()
    d = np.maximum(a,b)
    index = np.maximum(d,c)
    result = index.copy()
    result[index==a] = 0
    result[index==b] = 125
    result[index==c] = 255
    return t.from_numpy(result)
    
def get_label(label):
    
    label = label.numpy()
    label[label==0] = 0
    label[label==1] = 125
    label[label==2] = 255
    label = t.from_numpy(label)
    return label.float()

if __name__ == '__main__':

    train();    
     
