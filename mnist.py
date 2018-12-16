import os
import cv2
import sys
import random
import numpy as np

path = '/home/prithvi/dsets/MNIST/trainingSet/'

def load(train_count=3000,val_count=1000,one_hot=True):
    ds = []
    vds = []
    classes = os.listdir(path)
    unit = np.diag(np.ones(len(classes)))
    for n in os.listdir(path):
        n_path = os.path.join(path,n)
        if one_hot:
            lab = unit[int(n)]
        else:
            lab = int(n)
        flist = os.listdir(n_path)
        random.shuffle(flist)
        for s in flist[:train_count]:
            img = cv2.imread(os.path.join(n_path,s))
            img = np.float32(img)/255.
            ds.append((img,lab))
        for s in flist[train_count:train_count+val_count]:
            img = cv2.imread(os.path.join(n_path,s))
            img = np.float32(img)/255.
            vds.append((img,lab))
    #random.shuffle(ds)
    random.shuffle(vds)
    x,y = map(np.array,zip(*ds))
    vx,vy = map(np.array,zip(*vds))
    return (x,y,vx,vy)

if __name__ == '__main__':
    train_count = 3000
    val_count = 500
    x,y,vx,vy = load(train_count=train_count,val_count=val_count,one_hot=False)
    np.save('./mnist_'+str(train_count)+'_'+str(val_count),[x,y,vx,vy])


