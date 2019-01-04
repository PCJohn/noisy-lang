from __future__ import print_function
import sys
import cv2
import yaml
import numpy as np
import tensorflow as tf

import resnet_model

class Conv():
    def __init__(self,model='mnist'):
        self.model = model
        self.build_graph(model=model)

    def default_hparams(self):
        if self.model == 'mnist':
            niter = 6000
            print_iter = 200
            lr = 5e-4
            bz = 256
            beta1 = 0.5
            eps_t = 1e-2
        elif self.model == 'mnist_binary':
            niter = 4000
            print_iter = 200
            bz = 100
            lr = 5e-4
            beta1 = 0.5
            eps_t = 1e-2
        elif self.model == 'cifar10':
            niter = 6000
            print_iter = 500
            bz = 100
            lr = 1e-3
            beta1 = 0.5
            eps_t = 1e-2
        return {'niter':niter,'print_iter':print_iter,'lr':lr,'bz':bz,'beta1':beta1,'eps_t':eps_t}

    def build_graph(self,model='mnist'):
        xinit = tf.contrib.layers.xavier_initializer
        binit = tf.constant_initializer(0.0)
        relu = tf.nn.relu
        
        g = tf.get_default_graph()
        
        self.lr = tf.placeholder(tf.float32,name='lr')
        self.beta1 = tf.placeholder(tf.float32,name='beta1')
        self.eps_t = tf.placeholder(tf.float32,name='eps_t')
        self.trn_ph = tf.placeholder(tf.bool,name='train_ph')

        if model == 'mnist':
            self.x = tf.placeholder(tf.float32,shape=(None,28,28,1),name='input_ph')
            self.y = tf.placeholder(tf.int32,shape=(None,),name='label_ph')

            out = self.x
            out = tf.layers.conv2d(out,32,3,strides=2,activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
            out = tf.layers.conv2d(out,64,3,strides=2,activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
            out = tf.layers.conv2d(out,128,3,strides=2,activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
            out = tf.reshape(out,[-1,np.prod(out.get_shape().as_list()[1:])])
            out = tf.layers.dropout(out,rate=0.5,training=self.trn_ph)
        
            logits = tf.layers.dense(out,10,kernel_initializer=xinit(),bias_initializer=binit)
            self.pred = tf.argmax(logits,axis=1,name='pred_op')

        elif model == 'mnist_binary':
            self.x = tf.placeholder(tf.float32,shape=(None,28,28,1),name='input_ph')
            self.y = tf.placeholder(tf.int32,shape=(None,),name='label_ph')
                
            out = self.x
            out = tf.layers.conv2d(out,32,3,strides=2,activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
            out = tf.layers.conv2d(out,64,3,strides=2,activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
            out = tf.layers.conv2d(out,128,3,strides=2,activation=relu,padding='same',kernel_initializer=xinit(),bias_initializer=binit)
            out = tf.reshape(out,[-1,np.prod(out.get_shape().as_list()[1:])])
            out = tf.layers.dropout(out,rate=0.5,training=self.trn_ph)
                       
            logits = tf.layers.dense(out,2,kernel_initializer=xinit(),bias_initializer=binit)
            self.pred = tf.argmax(logits,axis=1,name='pred_op')
 
        elif model == 'cifar10':
            self.x = tf.placeholder(tf.float32,shape=(None,32,32,3),name='input_ph')
            self.y = tf.placeholder(tf.int32,shape=(None,),name='label_ph')
            resnet_size = 56
            num_blocks = (resnet_size - 2) // 6
            conv = resnet_model.Model(resnet_size=resnet_size,
                        bottleneck=False,
                        num_classes=10,
                        num_filters=16,
                        kernel_size=3,
                        conv_stride=1,
                        first_pool_size=None,
                        first_pool_stride=None,
                        block_sizes=[num_blocks] * 3,
                        block_strides=[1, 2, 2],
                        resnet_version=resnet_model.DEFAULT_VERSION,
                        data_format='channels_last')
        
            logits = conv(self.x,self.trn_ph)
            self.pred = tf.argmax(logits,axis=1,name='pred_op')
       
        loss = tf.losses.sparse_softmax_cross_entropy(logits=logits,labels=self.y)

        # adam opt update
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=tf.reshape(self.beta1,[]))
        self.adam_op = self.opt.minimize(loss)

        # langevin updates
        grads,varlist = list(zip(*self.opt.compute_gradients(loss)))
        grads = [ (g + tf.random_normal(g.get_shape().as_list(),mean=0.0,stddev=self.eps_t))
                    for g in grads]
        self.lang_op = self.opt.apply_gradients(list(zip(grads,varlist)))

    def train(self,sess,x,y,vx,vy,reset=False,update='adam',use_dropout=True,hparams={}):
        # fetch hyperparams
        def_hparams = self.default_hparams()
        hparams = dict([(p,hparams[p]) if (p in hparams) else (p,def_hparams[p]) for p in def_hparams])
        niter = hparams['niter']
        lr = hparams['lr']
        bz = hparams['bz']
        eps_t = hparams['eps_t']
        print_iter = hparams['print_iter']
        beta1 = hparams['beta1']
        # reset if needed
        if reset:
            sess.run(tf.global_variables_initializer(),feed_dict={self.beta1:beta1})
        
        val_t = [(0,0)]
        for itr in range(niter):
            bi = np.random.randint(0,x.shape[0],bz)
            bx,by = x[bi],y[bi]
            
            if update == 'adam':
                sess.run(self.adam_op,feed_dict={self.x:bx,self.y:by,self.trn_ph:use_dropout,self.lr:lr,self.beta1:beta1})
            elif update == 'langevin':
                sess.run(self.lang_op,feed_dict={self.x:bx,self.y:by,self.trn_ph:use_dropout,self.lr:lr,self.beta1:beta1,self.eps_t:eps_t})
            
            if itr%print_iter == 0:
                py = sess.run(self.pred,feed_dict={self.x:vx,self.trn_ph:False})
                vacc = np.mean(py==vy)
                val_t.append((itr+1,vacc))
                print('\tIteration:',itr,'\t',vacc)
        print('Final val:',val_t[-1][1])
        return val_t

    def save(self,path):
        tf.train.Saver().save(self.sess,path)

    def predict(self,sess,x):
        return sess.run(self.pred,feed_dict={self.x:x,self.trn_ph:False})

if __name__ == '__main__':
    #x,y,vx,vy = np.load('./mnist_3000_1000.npy')
    x,y,vx,vy = np.load('./mnist_3000_1000.npy')
    model_path = './models/mnist_model'
    
    with tf.Session() as sess:
        model = Conv()
        sess.run(tf.global_variables_initializer())
        if sys.argv[1] == 'adam':
            model.train(sess,x,noisy_y,vx,vy,update='adam')
            """# demo
            for i in range(5):
                py = sess.run(model.pred)
                cv2.imshow(str(py[i])+'--'+str(vy[i]),vx[i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            """
        elif sys.argv[1] == 'langevin':
            model.train(sess,x,noisy_y,vx,vy,update='langevin')
            #tf.train.Saver().save(sess,model_path)


