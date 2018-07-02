import cv2
import numpy as np
from PIL import Image
import os

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def getGray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(img)
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    equ = cv2.LUT(equ,table)
    return equ

def Crop(image,index,name,end):
    
    img = image.copy()
    bottom = img[0:256,0:256]
    left = img[256:512,0:256]
    top =img[0:256,256:512]
    right = img[256:512,256:512] 
    cv2.imwrite(name+str(index)+end,bottom)
    cv2.imwrite(name+str(index+1)+end,left)
    cv2.imwrite(name+str(index+2)+end,top)
    cv2.imwrite(name+str(index+3)+end,right)

    

def get_256():
    data = os.listdir('./data')
    index = 0
    for i in range(len(data)):
        data_path = data[i]
        label_path = (data[i].split('.')[0]).split('_')[0]+'_manual1.gif'
        data_img = cv2.imread('./data/'+data_path)
        label_img = Image.open('./labels/'+label_path)
        label_img = np.asarray(label_img)
        data_img = cv2.resize(data_img,(512,512),interpolation=cv2.INTER_CUBIC)
        data_img = getGray(data_img)
        label_img = cv2.resize(label_img,(512,512),interpolation=cv2.INTER_CUBIC)

        Crop(data_img,index,'./images_256/','.tif')
        Crop(label_img,index,'./label_256/','.tif')
        index += 4

def write_128(image,name,index):
    img = image.copy()
    img1 = img[0:128,0:128]
    img2 = img[128:256,0:128]
    img3 = img[0:128,128:256]
    img4 = img[128:256,128:256]
    cv2.imwrite(name+str(index)+'.tif',img1)
    cv2.imwrite(name+str(index+1)+'.tif',img2)
    cv2.imwrite(name+str(index+2)+'.tif',img3)
    cv2.imwrite(name+str(index+3)+'.tif',img4)

def get_128():
    index = 0
    for i in range(0,1100):
        data_img = cv2.imread('./images_256/'+str(i)+'.tif')
        label_img = cv2.imread('./label_256/'+str(i)+'.tif')
        write_128(data_img,'./images_128/',index)
        write_128(label_img,'./label_128/',index)
        index = index+4
def get_512():
    data = os.listdir('./data')
    index = 0
    for i in range(len(data)):
        data_path = data[i]
        label_path = (data[i].split('.')[0]).split('_')[0]+'_manual1.gif'
        data_img = cv2.imread('./data/'+data_path)
        data_img = cv2.resize(data_img,(512,512),interpolation=cv2.INTER_CUBIC)
        label_img = Image.open('./labels/'+label_path)
        label_img = np.asarray(label_img)
        label_img = cv2.resize(label_img,(512,512),interpolation=cv2.INTER_CUBIC)
        data_img = getGray(data_img)
        cv2.imwrite('./images_512/'+str(index)+'.tif',data_img)
        cv2.imwrite('./label_512/'+str(index)+'.tif',label_img)
        index +=1

#data_img = cv2.imread('./label_128/1.tif')
#data_img = cv2.cvtColor(data_img, cv2.COLOR_BGR2GRAY)
#print(data_img.shape)
#get_128()
#img = Image.open(path)
#img = img.resize((512,512))
#img = np.asarray(img)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print(img)
#equ = cv2.equalizeHist(img)
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#equ = clahe.apply(img)
#gamma = 1.2
#invGamma = 1.0 / gamma
#table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#equ = cv2.LUT(equ,table)
#img = Image.fromarray(equ)
#img.show()
#cv2.imshow('equ',data_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

get_512()
