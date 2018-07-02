import torch.nn as nn
import torch as t
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.autograd import Variable
from math import sqrt

class DownBlock(nn.Module):
    
    def __init__(self, in_channels=32, max_index=True, plus=True):
        super(DownBlock, self).__init__()
        self.max_index = max_index
        self.plus = plus
        self.conv1 = nn.Conv2d(in_channels,64,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2,stride=2)
        
        
    def forward(self,x):
        
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = x
        if self.plus:
            x = residual+x
        x = self.relu(x)
        if self.max_index:
            out = self.maxpool(x)
        return x,out
        
class UpBlock(nn.Module):
    
    def __init__(self, upsample_index=True, plus=True):
        super(UpBlock, self).__init__()
        self.upsample_index = upsample_index
        self.plus = plus
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1)
#        self.conv2 = nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, block):
        residual = x
        x = t.cat((block,x),1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.plus:
            x = residual+x
        x = self.relu(x)
        if self.upsample_index:
            x = self.upsample(x)
        return x
        


class DeepUNet(nn.Module):
    
    def __init__(self, colordim=1, downblock=DownBlock, upblock=UpBlock):
        super(DeepUNet, self).__init__()
        self.downlayer1 = downblock(colordim, plus=False)
        self.downlayer2 = downblock()
        self.downlayer3 = downblock()
        self.downlayer4 = downblock()
        self.downlayer5 = downblock()
        self.downlayer6 = downblock()
        self.downlayer7 = downblock(max_index=False)
        
        self.uplayer1 = upblock(plus=False)
        self.uplayer2 = upblock()
        self.uplayer3 = upblock()
        self.uplayer4 = upblock()
        self.uplayer5 = upblock()
        self.uplayer6 = upblock()
        self.uplayer7 = upblock(upsample_index=False)
        
        self.con1v1 = nn.Conv2d(32,1,kernel_size=1,stride=1,padding=0)
        self.bn_f = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        block1,x = self.downlayer1(x)
        block2,x = self.downlayer2(x)
        block3,x = self.downlayer3(x)
        block4,x = self.downlayer4(x)
        block5,x = self.downlayer5(x)
        block6,x = self.downlayer6(x)
        block7,x = self.downlayer7(x)
        
        x = self.uplayer1(x,block7)
        x = self.uplayer2(x,block6)
        x = self.uplayer3(x,block5)
        x = self.uplayer4(x,block4)
        x = self.uplayer5(x,block3)
        x = self.uplayer6(x,block2)
        x = self.uplayer7(x,block1)
        x = self.con1v1(x)
        x = self.bn_f(x)
        x = F.softsign(x)
        return x;
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
#def test():
#    net = DeepUNet()
#    net = net.cuda()
#    data = Image.open('21_training.tif')
#    data = data.resize((512,512))
#    trainsforms = T.Compose([T.ToTensor()])
#    data = trainsforms(data)
#    data = data.resize_(1,3,512,512)
#    data = Variable(data)
#    data = data.cuda()
#    out  = net(data)
#    print('result:')
##    
#test()

    
    