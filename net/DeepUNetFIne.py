import torch.nn as nn
import torch as t
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torch.autograd import Variable
from math import sqrt

class DownBlock(nn.Module):
    
    def __init__(self, in_channels,out_channels, max_index=True, plus=True):
        super(DownBlock, self).__init__()
        self.max_index = max_index
        self.plus = plus
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.downsample = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=1),nn.BatchNorm2d(out_channels))
        
    def forward(self,x):
        
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = x
        if self.plus:
            x = x+self.downsample(residual)
        x = self.relu(x)
        if self.max_index:
            out = self.maxpool(x)
        return x,out
        
class UpBlock(nn.Module):
    
    def __init__(self, in_channels,out_channels,upsample_index=True, plus=True):
        super(UpBlock, self).__init__()
        self.upsample_index = upsample_index
        self.plus = plus
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
#        self.conv2 = nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(nn.Conv2d(int(in_channels/2),out_channels,kernel_size=1),nn.BatchNorm2d(out_channels))

    def forward(self, x, block):
#        residual = x
        x = t.cat((block,x),1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
#        if self.plus:
#            x = x+self.downsample(residual)
        x = self.relu(x)
        if self.upsample_index:
            x = self.upsample(x)
        return x
        


class DeepUNetV2(nn.Module):
    
    def __init__(self, colordim=1, downblock=DownBlock, upblock=UpBlock):
        super(DeepUNetV2, self).__init__()
        self.downlayer1 = downblock(colordim,64, plus=False)
        self.downlayer2 = downblock(64,128)
        self.downlayer3 = downblock(128,256)
        self.downlayer4 = downblock(256,512)
        self.downlayer5 = downblock(512,1024,max_index=False)
        
        self.uplayer1 = upblock(2048,512,plus=False)
        self.uplayer2 = upblock(1024,256)
        self.uplayer3 = upblock(512,128)
        self.uplayer4 = upblock(256,64)
        self.uplayer5 = upblock(128,64,upsample_index=False)
        self.con1v1 = nn.Conv2d(64,3,kernel_size=1,stride=1,padding=0)
        self.bn_f = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.re_conv = nn.Sequential(nn.Conv2d(colordim,64,kernel_size=1,stride=1,padding=0),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        
    def forward(self, x):
        
#        residual = x
        block1,x = self.downlayer1(x)
        block2,x = self.downlayer2(x)
        block3,x = self.downlayer3(x)
        block4,x = self.downlayer4(x)
        block5,x = self.downlayer5(x)
 
        x = self.uplayer1(x,block5)
        x = self.uplayer2(x,block4)
        x = self.uplayer3(x,block3)
        x = self.uplayer4(x,block2)
        x = self.uplayer5(x,block1)
        
#        residual = self.re_conv(residual)
        x = self.con1v1(x)
#        x = self.bn_f(x)
#        x = self.relu(x)
#        x = F.softsign(x)
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

    
    