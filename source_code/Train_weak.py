## On the Texture Bias for Few-Shot CNN Segmentation, Implemented by Reza Azad ##
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import model as  M
import matplotlib.pyplot as plt
import utilz as U
import numpy as np
from parser_utils import get_parser
import pickle

## Get options
options = get_parser().parse_args()
t_l_path   = './fss_test_set.txt'
Best_performance = 0
Valid_miou = []

# Build the model
model = M.my_model(encoder = 'VGG_b345', input_size = (options.img_h, options.img_w, 3), k_shot = options.kshot, learning_rate = options.learning_rate)
model.summary()

# Load an episode of train
Train_list, Test_list = U.Get_tr_te_lists(options, t_l_path)


# Train on episodes
def train(opt):
    for ep in range(opt.epochs):
        epoch_loss = 0
        epoch_acc  = 0
        ## Get an episode for training model
        for idx in range(opt.iterations):
            support, smask, query, qmask = U.get_episode(opt, Train_list)
            acc_loss = model.train_on_batch([support, smask, query], qmask)
            epoch_loss += acc_loss[0]
            epoch_acc  += acc_loss[1]
            if (idx % 50) == 0:
                print ('Epoch>>',(ep+1),'>> Itteration:', (idx+1),'/',opt.iterations,' >>> Loss:', epoch_loss/(idx+1), ' >>> Acc:', epoch_acc/(idx+1))
        evaluate(opt, ep)

def evaluate(opt, ep):
    global Best_performance
    global Valid_miou
    overall_miou = 0.0
    for idx in range (opt.it_eval):
        ## Get an episode for evaluation 
        support, smask, query, qmask = U.get_episode_weakannotation(opt, Test_list)
        # Generate mask 
        Es_mask = model.predict([support, smask, query])
        # Compute MIOU for episode
        ep_miou       = U.compute_miou(Es_mask, qmask)
        overall_miou += ep_miou
    print('Epoch:', ep+1 ,'Validation miou >> ', (overall_miou / opt.it_eval))    
    # save model weights
    Valid_miou.append((overall_miou / opt.it_eval))
    if Best_performance<(overall_miou / opt.it_eval):
       Best_performance = (overall_miou / opt.it_eval)
       model.save_weights('fewshot_DOGLSTM_weak.h5')

def test(opt):
    model.load_weights('fewshot_DOGLSTM_weak.h5')
    overall_miou = 0.0
    for idx in range (opt.it_test):
        ## Get an episode for test 
        support, smask, query, qmask = U.get_episode(opt, Test_list)
        # Generate mask 
        Es_mask = model.predict([support, smask, query])
        # Compute MIOU for episode
        ep_miou       = U.compute_miou(Es_mask, qmask)
        overall_miou += ep_miou
        print('episode>>',(idx+1) ,'miou>>', ep_miou)
    print('Test miou >> ', (overall_miou / opt.it_test))    

train(options) 
test(options) 

Performance = {}
Performance['Valid_miou'] = Valid_miou

with open('performance_weak.pkl', 'wb') as f:
        pickle.dump(Performance, f, pickle.HIGHEST_PROTOCOL)

