import os
import cv2
import cPickle as pickle
import random
import numpy as np
from matplotlib import pyplot as plt

n_classes = 100
path = '/home/prithvi/dsets/cifar-100-python/'

def preproc(s):
    img = np.float32( np.transpose(np.reshape(s,(3,32,32)),(1,2,0))  )
    return img / 255.

def label(y):
    return np.diag(np.ones(n_classes))[y]

def load(split=1.0, one_hot=True):
    ds = []
    test = []
    for fname in os.listdir(path):
        if fname == 'train':
            f = open(os.path.join(path,fname), 'rb')
            tupled_data = pickle.load(f)
            f.close()
            print(tupled_data.keys())
            ds.extend(zip(tupled_data[b'data'],tupled_data[b'fine_labels']))
        elif fname == 'test':
            f = open(os.path.join(path,fname), 'rb')
            tupled_data = pickle.load(f)
            f.close()
            print(tupled_data.keys())
            test.extend(zip(tupled_data[b'data'],tupled_data[b'fine_labels']))

    random.shuffle(ds)
    split = int(len(ds)*split)
    train = ds[:split]

    print('>>>',len(train))
    x,y = zip(*train)
    tx,ty = zip(*test)

    x = np.array(map(preproc,x))
    tx = np.array(map(preproc,tx))
    if one_hot == True:
        y = map(label,y)
        ty = map(label,ty)
    y = np.array(y)
    ty = np.array(ty)

    print 'Cifar loaded'
    print 'Train shape:',x.shape,'--',y.shape
    print 'Test shape:',tx.shape,'--',ty.shape

    for i in range(30):
        s = np.uint8(255*x[i])
        plt.title(str(y[i]))
        plt.imshow(s)
        plt.show()
    return (x,y,tx,ty)

if __name__ == '__main__':
    tsplit = 1.0
    x,y,vx,vy = load(split=tsplit,one_hot=False)
    np.save('./cifar100_'+str(tsplit),[x,y,vx,vy])

    
