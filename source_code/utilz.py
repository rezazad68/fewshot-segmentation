## On the Texture Bias for Few-Shot CNN Segmentation, Implemented by Reza Azad ##
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import random 
import cv2
import matplotlib.pyplot as plt
import copy

## Generate Train and Test classes
def Get_tr_te_lists(opt, t_l_path):
    text_file = open(t_l_path, "r")
    Test_list = [x.strip() for x in text_file] 
    Class_list = os.listdir(opt.data_path)
    Train_list = []
    for idx in range(len(Class_list)):
        if not(Class_list[idx] in Test_list):
           Train_list.append(Class_list[idx])
    
    return Train_list, Test_list

def get_corner(X):
    corners = np.array([0, 0, 0, 0])
    corners[1] = X.shape[0]-1
    corners[3] = X.shape[1]-1
    while (np.sum(np.sum(X[corners[0], :, 0])))==0:
          corners[0] += 1
    while (np.sum(np.sum(X[corners[1] , :, 0])))==0:
          corners[1] -= 1
    while (np.sum(np.sum(X[:, corners[2], 0])))==0:
          corners[2] += 1
    while (np.sum(np.sum(X[:, corners[3], 0])))==0:
          corners[3] -= 1                    
    return  corners   

## Gen k-shot episode for query and support set
def get_episode(opt, setX):
    indx_c = random.sample(range(0, len(setX)), opt.nway)
    indx_s = random.sample(range(1, opt.class_samples+1), opt.class_samples)

    support = np.zeros([opt.nway, opt.kshot, opt.img_h, opt.img_w, 3], dtype = np.float32)
    smasks  = np.zeros([opt.nway, opt.kshot, 56,        56,        1], dtype = np.float32)
    query   = np.zeros([opt.nway,            opt.img_h, opt.img_w, 3], dtype = np.float32)      
    qmask   = np.zeros([opt.nway,            opt.img_h, opt.img_w, 1], dtype = np.float32)  
                
    for idx in range(len(indx_c)):
        for idy in range(opt.kshot): # For support set 
            s_img = cv2.imread(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy]) + '.jpg' )
            s_msk = cv2.imread(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy]) + '.png' )
            s_img = cv2.resize(s_img,(opt.img_h, opt.img_w))
            s_msk = cv2.resize(s_msk,(56,        56))        
            s_msk = s_msk /255.
            s_msk = np.where(s_msk > 0.5, 1., 0.)
            support[idx, idy] = s_img
            smasks[idx, idy]  = s_msk[:, :, 0:1] 
        for idy in range(1): # For query set consider 1 sample per class
            q_img = cv2.imread(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy+opt.kshot]) + '.jpg' )
            q_msk = cv2.imread(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy+opt.kshot]) + '.png' )
            q_img = cv2.resize(q_img,(opt.img_h, opt.img_w))
            q_msk = cv2.resize(q_msk,(opt.img_h, opt.img_w))        
            q_msk = q_msk /255.
            q_msk = np.where(q_msk > 0.5, 1., 0.)
            query[idx] = q_img
            qmask[idx] = q_msk[:, :, 0:1]        

    support = support /255.
    query   = query   /255.
   
    return support, smasks, query, qmask

## Gen k-shot episode for query and support set
def get_episode_weakannotation(opt, setX):
    indx_c = random.sample(range(0, len(setX)), opt.nway)
    indx_s = random.sample(range(1, opt.class_samples+1), opt.class_samples)

    support = np.zeros([opt.nway, opt.kshot, opt.img_h, opt.img_w, 3], dtype = np.float32)
    smasks  = np.zeros([opt.nway, opt.kshot, 56,        56,        1], dtype = np.float32)
    query   = np.zeros([opt.nway,            opt.img_h, opt.img_w, 3], dtype = np.float32)      
    qmask   = np.zeros([opt.nway,            opt.img_h, opt.img_w, 1], dtype = np.float32)  
                
    for idx in range(len(indx_c)):
        for idy in range(opt.kshot): # For support set 
            s_img = cv2.imread(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy]) + '.jpg' )
            s_msk = cv2.imread(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy]) + '.png' )
            cc = get_corner(s_msk)
            s_msk[cc[0]:cc[1], cc[2]:cc[3], :] = 255
            s_img = cv2.resize(s_img,(opt.img_h, opt.img_w))
            s_msk = cv2.resize(s_msk,(56,        56))        
            s_msk = s_msk /255.
            s_msk = np.where(s_msk > 0.5, 1., 0.)
            support[idx, idy] = s_img
            smasks[idx, idy]  = s_msk[:, :, 0:1] 
        for idy in range(1): # For query set consider 1 sample per class
            q_img = cv2.imread(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy+opt.kshot]) + '.jpg' )
            q_msk = cv2.imread(opt.data_path + setX[indx_c[idx]] + '/' + str(indx_s[idy+opt.kshot]) + '.png' )
            q_img = cv2.resize(q_img,(opt.img_h, opt.img_w))
            q_msk = cv2.resize(q_msk,(opt.img_h, opt.img_w))        
            q_msk = q_msk /255.
            q_msk = np.where(q_msk > 0.5, 1., 0.)
            query[idx] = q_img
            qmask[idx] = q_msk[:, :, 0:1]        

    support = support /255.
    query   = query   /255.
   
    return support, smasks, query, qmask
        
def compute_miou(Es_mask, qmask):
    ious = 0.0
    Es_mask = np.where(Es_mask> 0.5, 1. , 0.)
    for idx in range(Es_mask.shape[0]):
        notTrue = 1 -  qmask[idx]
        union = np.sum(qmask[idx] + (notTrue * Es_mask[idx]))
        intersection = np.sum(qmask[idx] * Es_mask[idx])
        ious += (intersection / union)
    miou = (ious / Es_mask.shape[0])
    return miou
    
    