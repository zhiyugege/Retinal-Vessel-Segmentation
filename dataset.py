from torch.utils import data
import numpy as np
from PIL import Image  
import torch
from torchvision import transforms as T
from config import DefaultConfig
from torch.utils.data import DataLoader


class DataSet(data.Dataset):  
    def __init__(self, root, transforms=None, train=True, test=False):
        
        self.image_folder = '/images_512/'
        self.label_folder = '/label_512/'
        self.txt_name = '/data_512.txt'
        self.root = root
        if train:
            roidbs = self.__prepare__('train')
            self.roidbs = roidbs
            
        if test:
            roidbs = self.__prepare__('test')
            self.roidbs = roidbs
        
        if test or train:
            self.trainsforms = T.Compose([
                    T.ToTensor(),
                ])
                
    def __getitem__(self, index): 
        
        roidb = self.roidbs[index]
        path = roidb['path']
        label_path = roidb['label']
        data = Image.open(path)
        data = self.enhance(data,roidb)
        data = self.trainsforms(data)

        label = Image.open(label_path)
        label = self.enhance(label,roidb)
        label = self.detailLabel(label)
        label = torch.from_numpy(label)
        return data,label
    
    def __len__(self):
        
        return len(self.roidbs)
    
    def __prepare__(self, name):
        
        print("prepare for "+name+" data....")
        root = self.root
        roidbs = []
        with open(root+name+self.txt_name) as f:
            for line in f:
                line = line.replace('\n','')
                path = root+name+self.image_folder+line+'.tif'
                label = root+name+self.label_folder+line+'.tif'
                img = {'path':path,
                       'label':label,
                       'flipedDegree':0,
                       'levelfliped':False,
                       'verticalfliped':False
                       }
                roidbs.append(img)   
                if name=='train':
                    roidbs = self.appendRoidbs(img, roidbs)


                
        print("preparing for "+name+" data is done!")        
        return roidbs
    def appendRoidbs(self, img, roidbs):
        
        r_img = img.copy()
        v_img = img.copy()
        r_img['levelfliped'] = True
        v_img['verticalfliped'] = True 
        roidbs.append(r_img)
        roidbs.append(v_img)
        for i in range(3,360,3):
            Image = img.copy()
            Image['flipedDegree'] = i
            r_Image = Image.copy()
            r_Image['levelfliped'] = True
            v_Image = Image.copy()
            v_Image['verticalfliped'] = True
            roidbs.append(Image)
            roidbs.append(r_Image)
            roidbs.append(v_Image)
        return roidbs
        
    
    def detailLabel(self,label):
        
        label = np.array(label)
        label[label==0]=0
        label[label==125]=1
        label[label==255]=2
        label = np.uint8(label)
        return label
    
    def enhance(self,img,roidb):
        
        degree = int(roidb['flipedDegree'])
        img = img.rotate(degree)
        if roidb['levelfliped']:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if roidb['verticalfliped']:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img
#train_data = DataSet('data/', train=True, test=False)
#print(len(train_data.roidbs))
#train_DataLoader =  DataLoader(train_data,1, shuffle=True, num_workers=4)
####i = 0
#for datas,label in train_DataLoader:
###    print(datas)
#    label = label.numpy()
#    print(len(label[label==122]))
#    print(np.unique(label))

    
    
        
