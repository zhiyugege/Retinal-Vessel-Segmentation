import os
import cv2
from PIL import Image
import numpy as np

def readImg(path):
    
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    
def showImg(img):
    cv2.imshow("Image", img)   
    cv2.waitKey (0)  
    cv2.destroyAllWindows()  
#    plt.imshow(img)
#    plt.show()
def findsim(img):
    
    pre = img.copy()
    for i in range(512):
        for j in range(512):
            if(pre[i][j]<=180 and pre[i][j]>=120):
                left = abs(pre[i-1][j]-pre[i][j])
                right = abs(pre[i+1][j]-pre[i][j])
                top = abs(pre[i][j+1]-pre[i][j])
                bottom = abs(pre[i][j-1]-pre[i][j])
                if max([left,right,top,bottom])>=15:
                    img[i][j] -= 50
#                if(cal(i,j,img,2))
#                    img[i][j]=125
    return img

def cal(i,j,img,value=2):
    if(inclinedR(i,j,img)<=value):
        return True
    if(inclinedL(i,j,img)<=value):
        return True
    if(transverse(i,j,img)<=value):
        return True
    if(portrait(i,j,img)<=value):
        return True
    return False

def inclinedR(row,col,img):
    
    x2 = 99
    x1 = -99
    i = row-1
    j = col-1
    while(j>=0 and i>=0):
        if img[i][j]==0:
            x1 = i
            break
        i -=1
        j -=1
    i = row+1
    j = col+1
    while(i<512 and j<512):
        if img[i][j]==0:
            x2 = i
            break
        i +=1
        j +=1
    return x2-x1

def inclinedL(row,col,img):
    
    x2 = 99
    x1 = -99
    i = row-1
    j = col+1
    while(j<512 and i>=0):
        if img[i][j]==0:
            x1 = i
            break
        i -=1
        j +=1
    i = row+1
    j = col-1
    while(i<512 and j>=0):
        if img[i][j]==0:
            x2 = i
            break
        i +=1
        j -=1
    return x2-x1
 
def transverse(row,col,img):
    
    x2 = 99
    x1 = -99
    for i in (col-1,-1,-1):
        if(img[row][i]==0):
            x1 = i
            break;
    for i in range(col+1,512):
        if(img[row][i]==0):
            x2 = i
            break;
    return x2-x1        

def portrait(row,col,img):
    
    y2 = 99
    y1 = -99
    for i in (row-1,-1,-1):
        if(img[i][col]==0):
            y1 = i
            break;
    for i in range(row+1,512):
        if(img[i][col]==0):
            y2 = i
            break;
    return y2-y1    

def create_new():
    
    img = readImg('./2/2_data.tif')
    img = findsim(img)
    showImg(img)
#    imgs = os.listdir('./old_img')
#    for img_name in imgs:
#        img = readImg('./old_img/'+img_name)
#        img[img<=125] = 0
#        img[img>125] = 255
#        img = findsim(img)
#        cv2.imwrite('./new_img/'+img_name,img)
create_new()
#img = Image.open('./new_img/0.tif')
#img = np.array(img)
#img = np.uint8(img)
#showImg(img)


#img = readImg('./new_img/0.tif')
#img[img==255] = 125
#print(img.shape)
#showImg(img)


