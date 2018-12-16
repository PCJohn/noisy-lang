from __future__ import print_function
import sys
import cv2
import numpy as np
import tensorflow as tf

class CifarConv():
    def __init__(self):
        self.niter = 6000
        self.print_iter = 500
        self.bz = 100
        self.lr = 1e-3
        self.beta1 = 0.5
        self.eps_t = 1e-2

        self.build_graph()

    def build_graph(self):
        self.x = tf.placeholder(tf.float32,shape=(None,32,32,3),name='input_ph')
        self.y = tf.placeholder(tf.int32,shape=(None,),name='label_ph')
        self.trn_ph = tf.placeholder(tf.bool,name='train_ph')

        xinit = tf.contrib.layers.xavier_initializer
        binit = tf.constant_initializer(0.0)
        relu = tf.nn.relu

        g = tf.get_default_graph()
        out = self.x
        out = tf.layers.conv2d(out,16,3,strides=(1,1),activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
        out = tf.layers.conv2d(out,16,3,strides=(1,1),activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
        out = tf.layers.conv2d(out,16,3,strides=(2,2),activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
        
        out = tf.layers.conv2d(out,32,3,strides=(1,1),activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
        out = tf.layers.conv2d(out,32,3,strides=(1,1),activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
        out = tf.layers.conv2d(out,32,3,strides=(2,2),activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
        
        out = tf.layers.conv2d(out,64,3,strides=(1,1),activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
        out = tf.layers.conv2d(out,64,3,strides=(1,1),activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
        out = tf.layers.conv2d(out,64,3,strides=(2,2),activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
        
        out = tf.layers.conv2d(out,128,3,strides=(1,1),activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
        out = tf.layers.conv2d(out,128,3,strides=(1,1),activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
        out = tf.layers.conv2d(out,128,3,strides=(2,2),activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
        
        out = tf.reshape(out,[-1,np.prod(out.get_shape().as_list()[1:])])
        out = tf.layers.dropout(out,rate=0.5,training=self.trn_ph)
        
        logits = tf.layers.dense(out,10,kernel_initializer=xinit(),bias_initializer=binit)
        self.pred = tf.argmax(logits,axis=1,name='pred_op')
        
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits,labels=self.y)

        # adam opt update
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=self.beta1)
        self.adam_op = self.opt.minimize(loss)

        # langevin updates
        grads,varlist = list(zip(*self.opt.compute_gradients(loss)))
        grads = [ (g + tf.random_normal(g.get_shape().as_list(),mean=0.0,stddev=self.eps_t))
                    for g in grads]
        self.lang_op = self.opt.apply_gradients(list(zip(grads,varlist)))

    def train(self,sess,x,y,vx,vy,update='adam',use_dropout=True):
        val_t = []
        for itr in range(self.niter):
            bi = np.random.randint(0,x.shape[0],self.bz)
            bx,by = x[bi],y[bi]
            
            if update == 'adam':
                sess.run(self.adam_op,feed_dict={self.x:bx,self.y:by,self.trn_ph:use_dropout})
            
            elif update == 'langevin':
                sess.run(self.lang_op,feed_dict={self.x:bx,self.y:by,self.trn_ph:use_dropout})
            
            if itr%self.print_iter == 0:
                py = sess.run(self.pred,feed_dict={self.x:vx,self.trn_ph:False})
                vacc = np.mean(py==vy)
                val_t.append((itr,vacc))
                print('\tIteration:',itr,'\t',vacc)
        print('Final val:',val_t[-1][1])
        return val_t

    def save(self,path):
        tf.train.Saver().save(self.sess,path)

    def predict(self,sess,x):
        return sess.run(self.pred,feed_dict={self.x:x,self.trn_ph:False})

if __name__ == '__main__':
    x,y,vx,vy = np.load('./cifar_0.7.npy')
    model_path = './models/cifar_model'
    
    with tf.Session() as sess:
        model = Conv()
        sess.run(tf.global_variables_initializer())
        
        model.train(sess,x,y,vx,vy,update='adam')


