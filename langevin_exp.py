from __future__ import print_function
import os
import sys
import cv2
import argparse
import numpy as np
import os.path as osp
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from conv import Conv

output_dir = './res'

def random_label_flip(y,p=0.0):
    all_val = set(y)
    rand = np.array([np.random.choice(list(all_val-set([y_]))) for y_ in y])
    flips = np.random.random(y.shape)
    noisy_y = y.copy()
    noisy_y[flips<p] = rand[flips<p]
    frac_correct = ((noisy_y==y).sum()/float(y.size))
    return noisy_y,frac_correct

def structured_label_flip(y,p=None):
    all_val = list(set(y))
    nclass = len(all_val)
    assert (p.shape[0] == p.shape[1]) and (p.shape[0] == len(all_val))
    if p is None:
        p = np.ones((nclass,nclass))/float(nclass) # default: uniform random flip
    noisy_y = np.array([np.random.choice(all_val,p=p[c]) for c in y]) # noise addition
    frac_correct = ((noisy_y==y).sum()/float(y.size)) # fraction of labels correct
    return noisy_y,frac_correct

'''
def structured_label_flip(y,T):
    all_val = set(y)
    flips = np.random.random(y.shape)
    noisy_y = y.copy()
    noisy_y = np.array([ np.argmax(T[lbl,:]==1-T[lbl,lbl]) if u < 1-T[lbl,lbl] \
                            else lbl for (lbl, u) in zip(y, flips) ])
    frac_correct = ((noisy_y==y).sum()/float(y.size))
    return noisy_y,frac_correct
'''

def noise_vs_val_acc(model,x,y,vx,vy):
    with tf.Session() as sess:
        conv = Conv(model=model)
        for update,use_dropout in [('adam',False),('langevin',False),('adam',True)]:
            title = str(update)
            if use_dropout:
                title += '+dropout'
            print('\n\nTraining with optimizer update: '+title)
            for p in [0, 0.5, 0.8, 0.9, 0.95, 1.0]:
                print('\n\nTraining with label noise p =',p,'\n')
                noisy_y,frac_correct = random_label_flip(y,p=p)
                print('Fraction of labels correct:',frac_correct,'\n\n')
                val_t = conv.train(sess,x,noisy_y,vx,vy,reset=True,update=update,use_dropout=use_dropout)
                itr_t,v_t = map(list,zip(*val_t))
                plt.plot(itr_t,v_t,label='p = '+str(p))
            plt.title('Update: '+title)
            plt.ylabel('Val acc.')
            plt.ylim((0,1))
            plt.xlabel('Iterations')
            lgd = plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
            plt.savefig(osp.join(output_dir,'noise_vs_val_acc_'+title+'.png'),bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.clf()

def structured_noise_exp(model,x,y,vx,vy):
    classlist = list(set(y))
    nclass = len(classlist)
    
    # manually set T
    def gen_noise_mat(label_noise_rate=0.5):
        T = np.eye(10)
        # T from Patrini et al, 2017
        T[2,2] = 1 - label_noise_rate
        T[2,7] = label_noise_rate
        T[3,3] = 1 - label_noise_rate
        T[3,8] = label_noise_rate
        T[5,5] = 1 - label_noise_rate
        T[5,6] = label_noise_rate
        T[6,6] = 1 - label_noise_rate
        T[6,5] = label_noise_rate
        T[7,7] = 1 - label_noise_rate
        T[7,1] = label_noise_rate
        """
        # T to simulate uniform random label flip
        T *= (1-label_noise_rate)
        T[T==0] = label_noise_rate/float(nclass-1)
        """
        return T
    
    with tf.Session() as sess:
        conv = Conv(model=model)
        for noise_level in [0.8]: #[0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            title = 'Noise level: '+str(noise_level)
            for update,use_dropout in [('adam',False),('langevin',False),('adam',True)]:
                label = str(update)
                if use_dropout:
                    label += '+dropout'
                for use_dither in [False,True]:
                    if use_dither:
                        label += '+dither'
                    print('\n\nTraining with optimizer update: '+str(update)+' dither: '+str(use_dither))
                    
                    T = gen_noise_mat(label_noise_rate=noise_level)
                    noisy_y,frac_correct = structured_label_flip(y,p=T)
                    print('Fraction of labels correct:',frac_correct,'\n\n')

                    print('Noise level:',noise_level,'\n\n')
                    print('Noise matrix:\n',T,'\n\n')
                
                    if use_dither:
                        D = np.eye(10)+0.5*np.random.random(size=T.shape) # dither matrix
                        D = D/(D.sum(axis=1)[:,np.newaxis])
                        print('Dither matrix:\n',D,'\n\n')
                        noisy_y,frac_correct = structured_label_flip(y,p=D) # apply dither
                
                    val_t = conv.train(sess,x,noisy_y,vx,vy,reset=True,update=update,use_dropout=use_dropout)
                    itr_t,v_t = map(list,zip(*val_t))
                    plt.plot(itr_t,v_t,label='Noise: '+str(label))
            plt.title(title)
            plt.ylabel('Val acc.')
            plt.ylim((0,1))
            plt.xlabel('Iterations')
            lgd = plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
            plt.savefig(osp.join(output_dir,'structured_exp_noise-'+str(noise_level)+'.png'),bbox_extra_artists=(lgd,),bbox_inches='tight')
            plt.clf()
            

def best_sgld_var_exp(model,x,y,vx,vy):
    with tf.Session() as sess:
        conv = Conv(model=model)
        best_sig_vs_p = []
        for p in np.arange(0,1,0.1):
            best_sig = 1e-4
            best_val = -1
            for ep in [-4,-3,-2,-1]:
                sig = 10**ep
                print('\n\np =',p,'sig =',sig,'\n')
                noisy_y,frac_correct = random_label_flip(y,p=p)
                
                val_t = conv.train(sess,x,noisy_y,vx,vy,reset=True,update='langevin',use_dropout=False,
                                    hparams={'eps_t':sig,'niter':4000})
                vacc = np.max([v[1] for v in val_t])
                if vacc > best_val:
                    best_sig = sig
                    best_val = vacc
            best_sig_vs_p.append((p,best_sig,best_val))
        
        # display table
        print('p\tbest_sig\tbest_val')
        for p,best_sig,best_val in best_sig_vs_p:
            print(p,'\t',best_sig,'\t',best_val)
        # plt variation
        p,best_sig,best_val = zip(*best_sig_vs_p)
        plt.plot(p,best_sig)
        plt.ylabel('Best sigma')
        plt.xlabel('Label noise level (p)')
        plt.savefig(osp.join(output_dir,'sigma_vs_p.png'),bbox_inches='tight')
        plt.clf()

def one_vs_all_exp(model,x,y,vx,vy):
    with tf.Session() as sess:
        conv = Conv(model=model)
        classlist = list(set(y))
        nclass = len(classlist)
        sc_list = [4] #[0,1,2,3,4]
        # single out a class in train and val sets: make labels in {0,1}
        gc_list = list(set(classlist)-set(sc_list))
        gy = y.copy()
        for sc in sc_list:
            gy[y==sc] = 0
        for gc in gc_list:
            gy[y==gc] = 1
        gvy = vy.copy()
        for sc in sc_list:
            gvy[vy==sc] = 0
        for gc in gc_list:
            gvy[vy==gc] = 1
        
        for update,use_dropout in [('adam',False),('langevin',False),('adam',True)]:
            title = str(update)
            if use_dropout:
                title += '+dropout'
            print('\n\nTraining with optimizer update: '+title)
            for p in [0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]:
                print('\n\nTraining with label noise p =',p,'\n')
                # generate noise transition matrix
                noisy_y,frac_correct = random_label_flip(gy,p=p)
                print('Fraction of labels correct:',frac_correct,'\n\n')
                val_t = conv.train(sess,x,noisy_y,vx,gvy,reset=True,update=update,use_dropout=use_dropout)
                itr_t,v_t = map(list,zip(*val_t))
                plt.plot(itr_t,v_t,label='p = '+str(p))
            plt.title('Update: '+title)
            plt.ylabel('Val acc.')
            plt.ylim((0,1))
            plt.xlabel('Iterations')
            lgd = plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
            plt.savefig(osp.join(output_dir,'one_vs_all_val_acc_'+title+'.png'),bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.clf()

def sgld_noise_level_vs_val_acc(model,x,y,vx,vy):
    with tf.Session() as sess:
        conv = Conv(model=model)
        use_dropout = False
        p = 0.8
        noisy_y,frac_correct = random_label_flip(y,p=p)
        print('\n\nUsing label noise p = '+str(p))
        print('Fraction of labels correct:',frac_correct,'\n\n')
        for lr in [1e-3,5e-4,1e-4,1e-3]:
            title = str(lr)
            for sgld_noise_level in [5e-1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
                print('\n\nlr =',lr,'SGLD noise =',sgld_noise_level,'\n')
                val_t = conv.train(sess,x,noisy_y,vx,vy,reset=True,update='langevin',use_dropout=False,
                                        hparams={'lr':lr,'eps_t':sgld_noise_level})
                itr_t,v_t = map(list,zip(*val_t))
                plt.plot(itr_t,v_t,label='sgld noise level: '+str(sgld_noise_level))
            # plot plain adam with this learning rate
            print('\n\nlr =',lr,'Update = Adam\n')
            val_t = conv.train(sess,x,noisy_y,vx,vy,reset=True,update='adam',use_dropout=False,hparams={'lr':lr})
            itr_t,v_t = map(list,zip(*val_t))
            plt.plot(itr_t,v_t,linestyle='--',label='adam')
            # plot adam+dropout with this learning rate
            print('\n\nlr =',lr,'Update = Adam+Dropout\n')
            val_t = conv.train(sess,x,noisy_y,vx,vy,reset=True,update='adam',use_dropout=True,hparams={'lr':lr})
            itr_t,v_t = map(list,zip(*val_t))
            plt.plot(itr_t,v_t,linestyle=':',label='adam+dropout')
            # format plot
            plt.title('Learning rate: '+title)
            plt.ylabel('Val acc.')
            plt.ylim((0,1))
            plt.xlabel('Iterations')
            lgd = plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
            plt.savefig(osp.join(output_dir,'sgld_noise_level_vs_val_acc_lr_'+title+'.png'),bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.clf()


if __name__ == '__main__':
    # load mnist
    x,y,vx,vy = np.load('./mnist_3000_500.npy',encoding='latin1')
    
    #model = 'mnist'
    #noise_vs_val_acc(model,x,y,vx,vy)
    
    #model = 'mnist'
    #best_sgld_var_exp(model,x,y,vx,vy)
    
    model = 'mnist'
    structured_noise_exp(model,x,y,vx,vy)

    #model = 'mnist_binary'
    #one_vs_all_exp(model,x,y,vx,vy)

    #model = 'mnist'
    #sgld_noise_level_vs_val_acc(model,x,y,vx,vy)

