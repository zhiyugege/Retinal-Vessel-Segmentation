import os

root = '../data/test/'
names = os.listdir(root+'images_512/')
index = 0
with open(root+'data_512.txt','w') as f:
    for name in names:
#        name = name.split('.')[0]
        f.write(str(index)+'\n')
        index = index+1       
