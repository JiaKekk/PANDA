#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
torch.set_default_tensor_type(torch.FloatTensor)
from torch.autograd import Variable
from torch.nn import functional as F
from torchmetrics import Accuracy, Recall, Precision, Specificity, ConfusionMatrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import matplotlib.pyplot as plt
from collections import Counter
import random
import math
import torch.optim as optim
from tabulate import tabulate
from ray import tune

from fractions import Fraction
from tqdm import tqdm


# In[ ]:


#MLP_prior


class MLP_prior(torch.nn.Module):
    def __init__(self,seed):
        super(MLP_prior, self).__init__()
        torch.manual_seed(seed)
        self.linear1 = torch.nn.Parameter(torch.randn(500))
        self.linear2 = torch.nn.Linear(500,22)
        self.linear3 = torch.nn.Linear(22,1)

    def forward(self, x):
        layer1_out = self.linear1*x
        layer2_out = F.relu(self.linear2(layer1_out))
        out= torch.sigmoid(self.linear3(layer2_out))
        return out, layer1_out, layer2_out


# In[ ]:


class MLP_final(torch.nn.Module):
    def __init__(self,seed):
        super(MLP_final, self).__init__()
        torch.manual_seed(seed)
        self.linear1 = torch.nn.Parameter(torch.randn(500))
        self.linear2 = torch.nn.Linear(500,22)
        self.linear3 = torch.nn.Linear(22,1)
    def forward(self, x):
        layer1_out = self.linear1*x
        layer2_out = F.relu(self.linear2(layer1_out))
        out = torch.sigmoid(self.linear3(layer2_out))
        return out, layer1_out, layer2_out

    
# In[ ]:


# Step 1: Assume the prior information is fully credible, only the covariates in $\mathcal{G}^p$ should be included.

def train_model_main_prior(config):
    inputs_1,targets_1 = config["inputs_1"],config["targets_1"]
    inputs_2,targets_2 = config["inputs_2"],config["targets_2"]
    inputs_3,targets_3 = config["inputs_3"],config["targets_3"]
    
   
    prior=config["prior"]
    
    ds_1_train_idx_list,ds_1_test_idx_list = config["ds_1_train_idx_list"],config["ds_1_test_idx_list"]
    ds_2_train_idx_list,ds_2_test_idx_list = config["ds_2_train_idx_list"],config["ds_2_test_idx_list"]
    ds_3_train_idx_list,ds_3_test_idx_list = config["ds_3_train_idx_list"],config["ds_3_test_idx_list"]
    
    split_num=5
    kf = KFold(n_splits=split_num)


    sum_test_BCE_loss=0
    for i in range(split_num):
        idx_train_1,idx_train_2,idx_train_3 = ds_1_train_idx_list[i],ds_2_train_idx_list[i],ds_3_train_idx_list[i]
        idx_test_1,idx_test_2,idx_test_3 = ds_1_test_idx_list[i],ds_2_test_idx_list[i],ds_3_test_idx_list[i]
        
        lambda1,lambda2,alpha,lr,ga= config["lambda1"], config["lambda2"],config["alpha"],config["lr"],config["ga"]
        model_1=MLP_prior(seed=1)
        model_2=MLP_prior(seed=1)
        model_3=MLP_prior(seed=1)
        max_iteration=1000
        learning_rate=lr
        params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
        optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
        loss_fn = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

        BCE=[]

        for t in range(max_iteration):
            optimizer.zero_grad() # renew optimizer
            out_1, layer1_out_1, layer2_out_1= model_1(inputs_1[idx_train_1])
            out_2, layer1_out_2, layer2_out_2= model_2(inputs_2[idx_train_2])
            out_3, layer1_out_3, layer2_out_3= model_3(inputs_3[idx_train_3])# forward propagate

            # extract parameters
            #[:-1] for leaving out bias term#
            model_1_all_linear1_params = model_1.linear1
            model_1_all_linear2_params= torch.cat([x.view(-1) for x in model_1.linear2.parameters()][:-1])
            model_1_all_linear3_params= torch.cat([x.view(-1) for x in model_1.linear3.parameters()][:-1])
           

            model_2_all_linear1_params = model_2.linear1
            model_2_all_linear2_params= torch.cat([x.view(-1) for x in model_2.linear2.parameters()][:-1])
            model_2_all_linear3_params= torch.cat([x.view(-1) for x in model_2.linear3.parameters()][:-1])
    

            model_3_all_linear1_params = model_3.linear1
            model_3_all_linear2_params= torch.cat([x.view(-1) for x in model_3.linear2.parameters()][:-1])
            model_3_all_linear3_params= torch.cat([x.view(-1) for x in model_3.linear3.parameters()][:-1])
    


            # compute loss
            all_linear1_params=torch.cat(( model_1_all_linear1_params.reshape(1,-1),  model_2_all_linear1_params.reshape(1,-1), model_3_all_linear1_params.reshape(1,-1)),0)
            all_linear23_params=torch.cat((model_1_all_linear2_params,model_1_all_linear3_params,
                                           model_2_all_linear2_params,model_2_all_linear3_params,
                                           model_3_all_linear2_params,model_3_all_linear3_params),0)

            non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)

            BCE_loss_1 = loss_fn(out_1, targets_1[idx_train_1].reshape(-1,1))
            BCE_loss_2 = loss_fn(out_2, targets_2[idx_train_2].reshape(-1,1))
            BCE_loss_3 = loss_fn(out_3, targets_3[idx_train_3].reshape(-1,1))


            l1_regularization = lambda1 * lambda2 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=1,dim=0))
            
            #L2
            l2_regularization = lambda1 *(1-lambda2)* torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

            #L2 
            l2_0_regularization = alpha * torch.sum(all_linear23_params.pow(2))
            
            BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
            loss =BCE_loss +l1_regularization +l2_regularization +l2_0_regularization

            # record loss
            BCE.append(BCE_loss.item())

            # compute derivative
            loss.backward()

            # gradient descent
            optimizer.step()

            # learning rate decay
            scheduler.step()
        
        
        test_out_1, test_layer1_out_1, test_layer2_out_1= model_1(inputs_1[idx_test_1])
        test_out_2, test_layer1_out_2, test_layer2_out_2= model_2(inputs_2[idx_test_2])
        test_out_3, test_layer1_out_3, test_layer2_out_3= model_3(inputs_3[idx_test_3])
        
        test_BCE_loss_1 = loss_fn(test_out_1, targets_1[idx_test_1].reshape(-1,1))
        test_BCE_loss_2 = loss_fn(test_out_2, targets_2[idx_test_2].reshape(-1,1))
        test_BCE_loss_3 = loss_fn(test_out_3, targets_3[idx_test_3].reshape(-1,1))
        
        test_BCE_loss=test_BCE_loss_1+test_BCE_loss_2+test_BCE_loss_3
        
        sum_test_BCE_loss+=test_BCE_loss.item()

    #print("loss: ",)
    tune.report(my_test_BCE_loss=sum_test_BCE_loss/split_num)


# In[ ]:


def main_prior_model(lambda1,lambda2,alpha,lr,ga,inputs_1,inputs_2,inputs_3,targets_1,targets_2,targets_3,prior):
    learning_rate=lr
    max_iteration=1000
    
    BCE=[]
    
    model_1=MLP_prior(seed=1)
    model_2=MLP_prior(seed=1)
    model_3=MLP_prior(seed=1)

    loss_fn = torch.nn.BCELoss()
    params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
    optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

    loss_record=[]
    
    for t in tqdm(range(max_iteration)):
    
        # renew optimizer
        optimizer.zero_grad()

        # forward propagate
        out_1, layer1_out_1, layer2_out_1= model_1(inputs_1)
        out_2, layer1_out_2, layer2_out_2= model_2(inputs_2)
        out_3, layer1_out_3, layer2_out_3= model_3(inputs_3)



        # extract parameters
        #[:-1] for leaving out bias term#
        model_1_all_linear1_params = model_1.linear1
        model_1_all_linear2_params= torch.cat([x.view(-1) for x in model_1.linear2.parameters()][:-1])
        model_1_all_linear3_params= torch.cat([x.view(-1) for x in model_1.linear3.parameters()][:-1])


        model_2_all_linear1_params = model_2.linear1
        model_2_all_linear2_params= torch.cat([x.view(-1) for x in model_2.linear2.parameters()][:-1])
        model_2_all_linear3_params= torch.cat([x.view(-1) for x in model_2.linear3.parameters()][:-1])


        model_3_all_linear1_params = model_3.linear1
        model_3_all_linear2_params= torch.cat([x.view(-1) for x in model_3.linear2.parameters()][:-1])
        model_3_all_linear3_params= torch.cat([x.view(-1) for x in model_3.linear3.parameters()][:-1])


        # compute loss

        all_linear1_params=torch.cat(( model_1_all_linear1_params.reshape(1,-1),  model_2_all_linear1_params.reshape(1,-1), model_3_all_linear1_params.reshape(1,-1)),0)
        all_linear23_params=torch.cat((model_1_all_linear2_params,model_1_all_linear3_params, 
                                         model_2_all_linear2_params,model_2_all_linear3_params,
                                         model_3_all_linear2_params,model_3_all_linear3_params),0)

        BCE_loss_1 = loss_fn(out_1, targets_1.reshape(-1,1))
        BCE_loss_2 = loss_fn(out_2, targets_2.reshape(-1,1))
        BCE_loss_3 = loss_fn(out_3, targets_3.reshape(-1,1))
        
        non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)

        #L1 
        l1_regularization = lambda1 * lambda2 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=1,dim=0))

        #L2 
        l2_regularization = lambda1 *(1-lambda2)* torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

        #L2 
        l2_0_regularization = alpha * torch.sum(all_linear23_params.pow(2))

        BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
        loss =BCE_loss +l1_regularization +l2_regularization +l2_0_regularization

        loss_record.append(loss.item())
        # record loss
        BCE.append(BCE_loss.item())

        # compute derivative
        loss.backward()

        # gradient descent
        optimizer.step()

        # learning rate decay
        scheduler.step()
    
################## Y_prior##################    
    y_prior_1= (model_1(inputs_1)[0]).detach().squeeze()

    y_prior_2= (model_2(inputs_2)[0]).detach().squeeze()

    y_prior_3= (model_3(inputs_3)[0]).detach().squeeze()
    
    return y_prior_1,y_prior_2,y_prior_3,model_1,model_2,model_3


# Step 2: Balance the current data information and the prior information.

# In[ ]:


def train_model_main_final(config):
    split_num=5
    kf = KFold(n_splits=split_num)
    max_iteration = 1000
    
    
    lambda1,lambda2,alpha,eta,lr,ga,= config["lambda1"], config["lambda2"],config["alpha"],config["eta"],config["lr"],config["ga"]
    inputs_1,train_y1 = config["inputs_1"],config["train_y1"]
    inputs_2,train_y2 = config["inputs_2"],config["train_y2"]
    inputs_3,train_y3 = config["inputs_3"],config["train_y3"]
    
    y_prior_1=config["y_prior_1"]
    y_prior_2=config["y_prior_2"]
    y_prior_3=config["y_prior_3"]
    prior=config["prior"]
    
    
    ds_1_train_idx_list,ds_1_test_idx_list = config["ds_1_train_idx_list"],config["ds_1_test_idx_list"]
    ds_2_train_idx_list,ds_2_test_idx_list = config["ds_2_train_idx_list"],config["ds_2_test_idx_list"]
    ds_3_train_idx_list,ds_3_test_idx_list = config["ds_3_train_idx_list"],config["ds_3_test_idx_list"]
    
    targets_1=(1-eta)*train_y1+eta*y_prior_1
    targets_2=(1-eta)*train_y2+eta*y_prior_2
    targets_3=(1-eta)*train_y3+eta*y_prior_3
    
    
    sum_tGM=0
    sum_test_BCE_loss=0
    for i in range(split_num):
        idx_train_1,idx_train_2,idx_train_3 = ds_1_train_idx_list[i],ds_2_train_idx_list[i],ds_3_train_idx_list[i]
        idx_test_1,idx_test_2,idx_test_3 = ds_1_test_idx_list[i],ds_2_test_idx_list[i],ds_3_test_idx_list[i]      
       
        model_1=MLP_final(seed=2)
        model_2=MLP_final(seed=155)
        model_3=MLP_final(seed=2605)
        
        
        max_iteration=1000
        learning_rate=lr
        params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
        optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
        loss_fn = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

        BCE=[]

        for t in range(max_iteration):
            optimizer.zero_grad() # renew optimizer

            out_1, layer1_out_1, layer2_out_1= model_1(inputs_1[idx_train_1])
            out_2, layer1_out_2, layer2_out_2= model_2(inputs_2[idx_train_2])
            out_3, layer1_out_3, layer2_out_3= model_3(inputs_3[idx_train_3])# forward propagate

            # extract parameters
            #[:-1] for leaving out bias term#
            model_1_all_linear1_params = model_1.linear1
            model_1_all_linear2_params= torch.cat([x.view(-1) for x in model_1.linear2.parameters()][:-1])
            model_1_all_linear3_params= torch.cat([x.view(-1) for x in model_1.linear3.parameters()][:-1])
           

            model_2_all_linear1_params = model_2.linear1
            model_2_all_linear2_params= torch.cat([x.view(-1) for x in model_2.linear2.parameters()][:-1])
            model_2_all_linear3_params= torch.cat([x.view(-1) for x in model_2.linear3.parameters()][:-1])
    

            model_3_all_linear1_params = model_3.linear1
            model_3_all_linear2_params= torch.cat([x.view(-1) for x in model_3.linear2.parameters()][:-1])
            model_3_all_linear3_params= torch.cat([x.view(-1) for x in model_3.linear3.parameters()][:-1])
    


            # compute loss
            all_linear1_params=torch.cat(( model_1_all_linear1_params.reshape(1,-1),  model_2_all_linear1_params.reshape(1,-1), model_3_all_linear1_params.reshape(1,-1)),0)
            all_linear234_params=torch.cat((model_1_all_linear2_params,model_1_all_linear3_params, 
                                             model_2_all_linear2_params,model_2_all_linear3_params,
                                             model_3_all_linear2_params,model_3_all_linear3_params),0)

            non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)

            BCE_loss_1 = loss_fn(out_1, targets_1[idx_train_1].reshape(-1,1))
            BCE_loss_2 = loss_fn(out_2, targets_2[idx_train_2].reshape(-1,1))
            BCE_loss_3 = loss_fn(out_3, targets_3[idx_train_3].reshape(-1,1))

            #L1 
            l1_regularization = lambda1 * lambda2 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=1,dim=0))
            
            #L2 
            l2_regularization = lambda1 *(1-lambda2)* torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

            #L2 
            l2_0_regularization = alpha * torch.sum(all_linear234_params.pow(2))
            
            BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
            loss =BCE_loss +l1_regularization +l2_regularization +l2_0_regularization

            # record loss
            BCE.append(BCE_loss.item())

            # compute derivative
            loss.backward()

            # gradient descent
            optimizer.step()

            # learning rate decay
            scheduler.step()
        
        
        test_out_1, test_layer1_out_1, test_layer2_out_1= model_1(inputs_1[idx_test_1])
        test_out_2, test_layer1_out_2, test_layer2_out_2= model_2(inputs_2[idx_test_2])
        test_out_3, test_layer1_out_3, test_layer2_out_3= model_3(inputs_3[idx_test_3])
    

        test_BCE_loss_1 = loss_fn(test_out_1, train_y1[idx_test_1].reshape(-1,1))
        test_BCE_loss_2 = loss_fn(test_out_2, train_y2[idx_test_2].reshape(-1,1))
        test_BCE_loss_3 = loss_fn(test_out_3, train_y3[idx_test_3].reshape(-1,1))
        
        test_BCE_loss=test_BCE_loss_1+test_BCE_loss_2+test_BCE_loss_3
        
        sum_test_BCE_loss+=test_BCE_loss.item()

    tune.report(my_test_BCE_loss=sum_test_BCE_loss/split_num)



# In[ ]:


def main_final_model(lambda1,lambda2,alpha,lr,ga,eta,inputs_1,inputs_2,inputs_3,train_y1,train_y2,train_y3,y_prior_1,y_prior_2,y_prior_3,
                     test_inputs_1,test_inputs_2,test_inputs_3,test_targets_1,test_targets_2,test_targets_3,prior):
    learning_rate=lr
    max_iteration=1000

    # record loss descent
    BCE=[]
    targets_1=(1-eta)*train_y1+eta*y_prior_1
    targets_2=(1-eta)*train_y2+eta*y_prior_2
    targets_3=(1-eta)*train_y3+eta*y_prior_3

    # main nn object



    model_1=MLP_final(seed=2)
    model_2=MLP_final(seed=155)
    model_3=MLP_final(seed=2605)

    loss_fn = torch.nn.BCELoss()
    params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
    optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)

    # learning rate decay scheme
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

    loss_record=[]
    # loop for max_iteration times
    for t in tqdm(range(max_iteration)):

        # renew optimizer
        optimizer.zero_grad()

        # forward propagate

        out_1, layer1_out_1, layer2_out_1= model_1(inputs_1)
        out_2, layer1_out_2, layer2_out_2= model_2(inputs_2)
        out_3, layer1_out_3, layer2_out_3= model_3(inputs_3)



        # extract parameters
        #[:-1] for leaving out bias term#
        model_1_all_linear1_params = model_1.linear1
        model_1_all_linear2_params= torch.cat([x.view(-1) for x in model_1.linear2.parameters()][:-1])
        model_1_all_linear3_params= torch.cat([x.view(-1) for x in model_1.linear3.parameters()][:-1])


        model_2_all_linear1_params = model_2.linear1
        model_2_all_linear2_params= torch.cat([x.view(-1) for x in model_2.linear2.parameters()][:-1])
        model_2_all_linear3_params= torch.cat([x.view(-1) for x in model_2.linear3.parameters()][:-1])

        model_3_all_linear1_params = model_3.linear1
        model_3_all_linear2_params= torch.cat([x.view(-1) for x in model_3.linear2.parameters()][:-1])
        model_3_all_linear3_params= torch.cat([x.view(-1) for x in model_3.linear3.parameters()][:-1])


        # compute loss

        all_linear1_params=torch.cat(( model_1_all_linear1_params.reshape(1,-1),  model_2_all_linear1_params.reshape(1,-1), model_3_all_linear1_params.reshape(1,-1)),0)
        all_linear234_params=torch.cat((model_1_all_linear2_params,model_1_all_linear3_params, 
                                         model_2_all_linear2_params,model_2_all_linear3_params, 
                                         model_3_all_linear2_params,model_3_all_linear3_params),0)

        BCE_loss_1 = loss_fn(out_1, targets_1.reshape(-1,1))
        BCE_loss_2 = loss_fn(out_2, targets_2.reshape(-1,1))
        BCE_loss_3 = loss_fn(out_3, targets_3.reshape(-1,1))

        non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)


                   
        #L1 
        l1_regularization = lambda1 * lambda2 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=1,dim=0))

        #L2 
        l2_regularization = lambda1 *(1-lambda2)* torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

        #L2 
        l2_0_regularization = alpha * torch.sum(all_linear234_params.pow(2))

        BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
        loss =BCE_loss +l1_regularization +l2_regularization +l2_0_regularization

        loss_record.append(loss.item())
        # record loss
        BCE.append(BCE_loss.item())

        # compute derivative
        loss.backward()

        # gradient descent
        optimizer.step()

        # learning rate decay
        scheduler.step()
    ####################################   prediction ####################################      
    prediction_1= (model_1(test_inputs_1)[0]>0.5).clone().int()
    target_1= test_targets_1.reshape(-1,1).int()

    prediction_2= (model_2(test_inputs_2)[0]>0.5).clone().int()
    target_2= test_targets_2.reshape(-1,1).int()

    prediction_3= (model_3(test_inputs_3)[0]>0.5).clone().int()
    target_3= test_targets_3.reshape(-1,1).int()

    prediction=np.append(prediction_1,prediction_2)
    prediction=np.append(prediction,prediction_3)
    prediction=prediction.tolist()

    target=np.append(target_1,target_2)
    target=np.append(target,target_3)
    target=target.tolist()
    
    ####################################   variable selection ####################################     
    model_1_weight=model_1_all_linear1_params.detach().numpy().copy().reshape(1,-1)
    model_2_weight=model_2_all_linear1_params.detach().numpy().copy().reshape(1,-1)
    model_3_weight=model_3_all_linear1_params.detach().numpy().copy().reshape(1,-1)
    
    return prediction,target,model_1_weight,model_2_weight,model_3_weight






