import os
import cv2
import cPickle as pickle
import random
import numpy as np
from matplotlib import pyplot as plt

n_classes = 10
path = '/home/prithvi/dsets/cifar-10-batches-py/'

def preproc(s):
    img = np.float32( np.transpose(np.reshape(s,(3,32,32)),(1,2,0))  )
    return img / 255.

def label(y):
    return np.diag(np.ones(n_classes))[y]

def load(split=0.7, one_hot=True):
    ds = []
    test = []
    for fname in os.listdir(path):
        if fname.startswith('data_batch_'):
            f = open(os.path.join(path,fname), 'rb')
            tupled_data = pickle.load(f)
            f.close()
            ds.extend(zip(tupled_data[b'data'],tupled_data[b'labels']))
        elif fname == 'test_batch':
            f = open(os.path.join(path,fname), 'rb')
            tupled_data = pickle.load(f)
            f.close()
            test.extend(zip(tupled_data[b'data'],tupled_data[b'labels']))

    random.shuffle(ds)
    split = int(len(ds)*split)
    train = ds[:split]
    #val = ds[split:]

    x,y = zip(*train)
    tx,ty = zip(*test)
    #vx,vy = zip(*val)

    x = np.array(map(preproc,x))
    tx = np.array(map(preproc,tx))
    #vx = np.array(map(preproc,vx))
    if one_hot == True:
        y = map(label,y)
        ty = map(label,ty)
        #vy = map(label,vy)
    y = np.array(y)
    ty = np.array(ty)
    #vy = np.array(vy)

    print 'Cifar loaded'
    print 'Train shape:',x.shape,'--',y.shape
    #print 'Validation shape:',vx.shape,'--',vy.shape
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
    np.save('./cifar_'+str(tsplit),[x,y,vx,vy])

    
