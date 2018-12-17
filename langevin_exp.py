from __future__ import print_function
import os
import sys
import cv2
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
    noisy_y = np.array([np.random.choice(all_val,p=p[c]) for c in y])
    frac_correct = ((noisy_y==y).sum()/float(y.size)) # fraction of labels correct
    #row_ent = -(p*np.log(p+1e-8)).sum(axis=1) # entropy of each row
    #avg_ent = np.mean(row_ent) # average entropy of transition at each node
    noise_level = 0 # TODO
    return noisy_y,frac_correct,noise_level

def noise_vs_val_acc(model,x,y,vx,vy):
    with tf.Session() as sess:
        conv = Conv(model=model)
        #use_dropout = True
        for update,use_dropout in [('adam',False),('langevin',False),('adam',True)]:
            title = str(update)
            if use_dropout:
                title += '+dropout'
            print('\n\nTraining with optimizer update: '+title)
            for p in [0, 0.5, 0.8, 0.9, 0.95, 1.0]:
                print('\n\nTraining with label noise p =',p,'\n')
                noisy_y,frac_correct = random_label_flip(y,p=p)
                print('Fraction of labels correct:',frac_correct,'\n\n')
                sess.run(tf.global_variables_initializer()) # reset model
                val_t = conv.train(sess,x,noisy_y,vx,vy,update=update,use_dropout=use_dropout)
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
    with tf.Session() as sess:
        conv = Conv(model=model)
        #use_dropout = True
        classlist = list(set(y))
        nclass = len(classlist)
        for update,use_dropout in [('adam',False),('langevin',False),('adam',True)]:
            title = str(update)
            if use_dropout:
                title += '+dropout'
            print('\n\nTraining with optimizer update: '+title)
            nrounds = 100
            for _ in range(nrounds):
                p = np.random.random((nclass,nclass)) #noise transition matrix
                #p = np.diag(np.ones(nclass))
                #p = np.ones((nclass,nclass))/float(nclass)
                p /= p.sum(axis=1)[:,np.newaxis]
                noisy_y,frac_correct,avg_ent = structured_label_flip(y,p=p)
                print('Fraction of labels correct:',frac_correct,'\n\n')
                print('Mean transition entropy:',avg_ent,'\n\n')
                #sess.run(tf.global_variables_initializer()) # reset model
                #val_t = conv.train(sess,x,noisy_y,vx,vy,update=update,use_dropout=use_dropout)
                #itr_t,v_t = map(list,zip(*val_t))
                #plt.plot(itr_t,v_t,label='E = '+str(avg_ent))
            """plt.title('Update: '+title)
            plt.ylabel('Val acc.')
            plt.ylim((0,1))
            plt.xlabel('Iterations')
            lgd = plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
            plt.savefig(osp.join(output_dir,'noise_vs_val_acc_'+title+'.png'),bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.clf()
            """

def best_sgld_var_exp(model,x,y,vx,vy):
    with tf.Session() as sess:
        conv = Conv(model=model)
        best_sig_vs_p = []
        for p in np.arange(0,1,0.1):
            best_sig = 1e-4
            best_val = -1
            for ep in [-6,-5,-4,-3,-2,-1]:
                sig = 10**ep
                print('\n\np =',p,'sig =',sig,'\n')
                noisy_y,frac_correct = random_label_flip(y,p=p)
                
                sess.run(tf.global_variables_initializer()) # reset model
                conv.niter = 2000
                conv.eps_t = sig
                
                val_t = conv.train(sess,x,noisy_y,vx,vy,update='langevin',use_dropout=False)
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
                sess.run(tf.global_variables_initializer()) # reset model
                val_t = conv.train(sess,x,noisy_y,vx,gvy,update=update,use_dropout=use_dropout)
                itr_t,v_t = map(list,zip(*val_t))
                plt.plot(itr_t,v_t,label='p = '+str(p))
            plt.title('Update: '+title)
            plt.ylabel('Val acc.')
            plt.ylim((0,1))
            plt.xlabel('Iterations')
            lgd = plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
            plt.savefig(osp.join(output_dir,'one_vs_all_val_acc_'+title+'.png'),bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.clf()



if __name__ == '__main__':
    # load mnist
    x,y,vx,vy = np.load('./mnist_3000_500.npy',encoding='latin1')
    
    #model = 'mnist'
    #noise_vs_val_acc(model,x,y,vx,vy)
    
    #model = 'mnist'
    #best_sgld_var_exp(model,x,y,vx,vy)
    
    #model = 'mnist'
    #structured_noise_exp(model,x,y,vx,vy)

    model = 'mnist_binary'
    one_vs_all_exp(model,x,y,vx,vy)
