from sklearn import metrics
import cv2
import os
import numpy as np
from PIL import Image
import pylab as pl



def change(path,name,view=125):
    
    mask = np.array(Image.open('./mask/'+str(name+1)+'_test_mask.gif')).astype(np.uint8)

    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask,(512,512),interpolation=cv2.INTER_CUBIC)
    
    img[img<view] = 0
    img[img>=view] = 1
    img = img[mask==255].flatten()
#    img = img.reshape(-1)
    return img


def single(i):
        
    max_pre = []
    max_label = []
    f1_max = 0
    j_p = 0
    for j in range(0,255):
        label = change('./label/'+str(i)+'.tif',i,j)
        result = change('./true_best/'+str(i)+'.tif',i)
        f1 = metrics.f1_score(label, result) 
        if(f1>f1_max):
            f1_max = f1
            max_pre = result.copy()
            max_label = label.copy()
            j_p = j
    print(str(i)+'.tif',f1_max,j_p)
    return f1_max,max_pre,max_label
        

def f1_func(i):
    label = Image.open('./label/'+str(i)+'.tif').convert('1')
    label = np.array(label)
    label[label==False] = 0
    label[label==True] = 1
    label = label.reshape(-1)
    
    img = cv2.cvtColor(cv2.imread('./0.812/'+str(i)+'.tif'), cv2.COLOR_BGR2GRAY)
    pre = img.copy()
    result = img.reshape(-1)
#    return label,result
    
#    precision, recall, thresholds = metrics.precision_recall_curve(label,result,pos_label=1)
#    max_f1 = 0
#    best_view = 0
#    for index in range(len(precision)):
#        curr_f1 = 2. * precision[index] * recall[index] / (precision[index] + recall[index]+0.00000000001)
#        if(max_f1<curr_f1):
#            max_f1 = curr_f1
#            best_view = thresholds[index]
#    pre[img<best_view] = 0
#    pre[img>=best_view] = 255
##    cv2.imwrite('0.812_two/'+str(i)+'.tif',pre)
#    print(str(i)+'.tif',max_f1,best_view)
    return max_f1,result,label

#f = f1_func(15)
#f1 = single(2)

#pre_all = []
#label_all = []
#for i in range(0,20):
#    f1,pre,label = single(i)
#    pre_all = np.append(pre_all,pre)
#    label_all = np.append(label_all,label)
#precision, recall, thresholds = metrics.precision_recall_curve(label_all,pre_all,pos_label=1)
#test_auc = metrics.roc_auc_score(label_all,pre_all)
#print(precision,recall,test_auc)
#f1_all = metrics.f1_score(label_all, pre_all)
#fpr,tpr,thresholds_roc = metrics.roc_curve(label_all,pre_all)
#test_auc = metrics.roc_auc_score(label_all,pre_all)
#print(f1_all,test_auc)
   
#f1 = 0.826

def draw_pr():    
    
    pre_all = []
    label_all = []
    for i in range(0,20):       
        max_f1,pre,label = f1_func(i)    
        pre_all = np.append(pre_all,pre)
        label_all = np.append(label_all,label)
    precision, recall, thresholds = metrics.precision_recall_curve(label_all,pre_all,pos_label=1)
    test_auc = metrics.roc_auc_score(label_all,pre_all)
    print(test_auc)
    fpr,tpr,thresholds_roc = metrics.roc_curve(label_all,pre_all)
    print(len(fpr),len(precision))
    fpr_pl = []
    tpr_pl = []
    max_f1 = 0
    best_view = 0
    for index in range(len(precision)):
        fpr_pl.append(fpr[index])
        tpr_pl.append(tpr[index])
#        print(fpr[index]+tpr[index]-1)
        curr_f1 = 2. * precision[index] * recall[index] / (precision[index] + recall[index]+0.00000000001)
        if(max_f1<curr_f1):
            max_f1 = curr_f1
            best_view = thresholds[index]
    print(max_f1)
    pl.title("Precision Recall Curve")
    
    pl.xlabel("Recall")
    pl.ylabel("Precision")
    pl.xlim(0.5, 1.0)
    pl.ylim(0.5, 1.0)
    pl.plot(recall ,precision,label='DenseNet-binary')
    pl.legend(loc='lower left')
    pl.plot(0.803,0.849,'r*',label='binary+multi')   
    pl.legend(loc='lower left')
    pl.plot(0.785,0.839,'g+',label='DenseNet-multi')   
    pl.legend(loc='lower left')
#    pl.title("ROC Curve")
#    pl.xlim(0, 0.3)
#    pl.ylim(0.7, 1.0)
#    pl.plot(fpr , tpr ,label='DenseNet-binary-roc')
#    pl.legend(loc='lower right')
#    pl.show()

draw_pr()


    
    
    