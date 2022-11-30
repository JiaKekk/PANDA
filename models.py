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


# In[ ]:


#MLP_prior

#p=100
class MLP_prior(torch.nn.Module):
    def __init__(self,seed):
        super(MLP_prior, self).__init__()
        torch.manual_seed(seed)
        self.linear1 = torch.nn.Parameter(torch.randn(100))
        self.linear2 = torch.nn.Linear(100,10)
        self.linear3 = torch.nn.Linear(10,1)

    def forward(self, x):
        layer1_out = self.linear1*x
        layer2_out = F.relu(self.linear2(layer1_out))
        out= torch.sigmoid(self.linear3(layer2_out))
        return out, layer1_out, layer2_out


# In[ ]:


#MLP_final
class MLP_final(torch.nn.Module):
    def __init__(self,seed):
        super(MLP_final, self).__init__()
        torch.manual_seed(seed)
        self.linear1 = torch.nn.Parameter(torch.randn(100))
        self.linear2 = torch.nn.Linear(100,10)
        self.linear3 = torch.nn.Linear(10,10)
        self.linear4 = torch.nn.Linear(10,1)
    def forward(self, x):
        layer1_out = self.linear1*x
        layer2_out = F.relu(self.linear2(layer1_out))
        layer3_out = F.relu(self.linear3(layer2_out))
        out= torch.sigmoid(self.linear4(layer3_out))
        return out, layer1_out, layer2_out,layer3_out
    

# In[ ]:



class MLP_inte(torch.nn.Module):
        def __init__(self,seed):
            super(MLP_inte, self).__init__()
            torch.manual_seed(seed)
            self.linear1= torch.nn.Linear(100,1)

        def forward(self, x):
            out = torch.sigmoid(self.linear1(x))
            return out
# ## M1_prior

# In[ ]:


# main_prior
def train_model_main_prior(config):
    inputs_1,targets_1 = config["inputs_1"],config["targets_1"]
    inputs_2,targets_2 = config["inputs_2"],config["targets_2"]
    inputs_3,targets_3 = config["inputs_3"],config["targets_3"]
    
    prior=config["prior"]
    
    split_num=5
    kf = KFold(n_splits=split_num)

    sum_test_BCE_loss=0
    for idx_train,idx_test in kf.split(inputs_1):
        lambda1,lambda2,lr,ga= config["lambda1"], config["lambda2"],config["lr"],config["ga"]
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
            out_1, layer1_out_1, layer2_out_1= model_1(inputs_1[idx_train])
            out_2, layer1_out_2, layer2_out_2= model_2(inputs_2[idx_train])
            out_3, layer1_out_3, layer2_out_3= model_3(inputs_3[idx_train])# forward propagate

            # extract parameters
            #[:-1] for leaving out bias term#
            model_1_all_linear1_params = model_1.linear1
            model_1_all_linear2_params= torch.cat([x.view(-1) for x in model_1.linear2.parameters()][:-1])
            model_1_all_linear3_params= torch.cat([x.view(-1) for x in model_1.linear3.parameters()][:-1])
           

            model_2_all_linear1_params = model_1.linear1
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

            BCE_loss_1 = loss_fn(out_1, targets_1[idx_train].reshape(-1,1))
            BCE_loss_2 = loss_fn(out_2, targets_2[idx_train].reshape(-1,1))
            BCE_loss_3 = loss_fn(out_3, targets_3[idx_train].reshape(-1,1))

#             l1_regularization = lambda1 * smooth_l1(all_linear1_params,prior)
            l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

            l2_regularization = lambda2* torch.sum(all_linear23_params.pow(2))
            BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
            loss =BCE_loss+l1_regularization +l2_regularization

            # record loss
            BCE.append(BCE_loss.item())

            # compute derivative
            loss.backward()

            # gradient descent
            optimizer.step()

            # learning rate decay
            scheduler.step()
        
        
        test_out_1, test_layer1_out_1, test_layer2_out_1= model_1(inputs_1[idx_test])
        test_out_2, test_layer1_out_2, test_layer2_out_2= model_2(inputs_2[idx_test])
        test_out_3, test_layer1_out_3, test_layer2_out_3= model_3(inputs_3[idx_test])
        
        test_BCE_loss_1 = loss_fn(test_out_1, targets_1[idx_test].reshape(-1,1))
        test_BCE_loss_2 = loss_fn(test_out_2, targets_2[idx_test].reshape(-1,1))
        test_BCE_loss_3 = loss_fn(test_out_3, targets_3[idx_test].reshape(-1,1))
        
        test_BCE_loss=test_BCE_loss_1+test_BCE_loss_2+test_BCE_loss_3
        
        sum_test_BCE_loss+=test_BCE_loss.item()

    #print("loss: ",)
    tune.report(my_test_BCE_loss=sum_test_BCE_loss/split_num)


# In[ ]:


def main_prior_model(lambda1,lambda2,lr,ga,inputs_1,inputs_2,inputs_3,targets_1,targets_2,targets_3,prior):
    learning_rate=lr
    max_iteration=1000
    
    BCE=[]
    
    model_1=MLP_prior(seed=1)
    model_2=MLP_prior(seed=2)
    model_3=MLP_prior(seed=1)

    loss_fn = torch.nn.BCELoss()
    params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
    optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

    loss_record=[]
    
    for t in range(max_iteration):
    
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


        model_2_all_linear1_params = model_1.linear1
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
    #     l1_regularization = lambda1 * smooth_l1(all_linear1_params,prior)

        l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

        l2_regularization = lambda2 * torch.sum(all_linear23_params.pow(2))
        BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
        loss =BCE_loss+l1_regularization +l2_regularization

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
    
    return y_prior_1,y_prior_2,y_prior_3


# ## M1_final

# In[ ]:


def train_model_main_final(config):
    split_num=5
    kf = KFold(n_splits=split_num)
    
    
    lambda1,lambda2,eta,lr,ga,= config["lambda1"], config["lambda2"],config["eta"],config["lr"],config["ga"]
    inputs_1,train_y1 = config["inputs_1"],config["train_y1"]
    inputs_2,train_y2 = config["inputs_2"],config["train_y2"]
    inputs_3,train_y3 = config["inputs_3"],config["train_y3"]
    
    y_prior_1=config["y_prior_1"]
    y_prior_2=config["y_prior_2"]
    y_prior_3=config["y_prior_3"]
    prior=config["prior"]
    
    targets_1=(1-eta)*train_y1+eta*y_prior_1
    targets_2=(1-eta)*train_y2+eta*y_prior_2
    targets_3=(1-eta)*train_y3+eta*y_prior_3
    
    
    sum_tGM=0
    for idx_train,idx_test in kf.split(inputs_1):
       
        
        model_1=MLP_final(seed=1)
        model_2=MLP_final(seed=2)
        model_3=MLP_final(seed=1)
        max_iteration=1000
        learning_rate=lr
        params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
        optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
        loss_fn = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

        BCE=[]

        for t in range(max_iteration):
            optimizer.zero_grad() # renew optimizer
            out_1, layer1_out_1, layer2_out_1,layer3_out_1= model_1(inputs_1[idx_train])
            out_2, layer1_out_2, layer2_out_2,layer3_out_2= model_2(inputs_2[idx_train])
            out_3, layer1_out_3, layer2_out_3,layer3_out_3= model_3(inputs_3[idx_train])# forward propagate

            # extract parameters
            #[:-1] for leaving out bias term#
            model_1_all_linear1_params = model_1.linear1
            model_1_all_linear2_params= torch.cat([x.view(-1) for x in model_1.linear2.parameters()][:-1])
            model_1_all_linear3_params= torch.cat([x.view(-1) for x in model_1.linear3.parameters()][:-1])
            model_1_all_linear4_params= torch.cat([x.view(-1) for x in model_1.linear4.parameters()][:-1])
           

            model_2_all_linear1_params = model_1.linear1
            model_2_all_linear2_params= torch.cat([x.view(-1) for x in model_2.linear2.parameters()][:-1])
            model_2_all_linear3_params= torch.cat([x.view(-1) for x in model_2.linear3.parameters()][:-1])
            model_2_all_linear4_params= torch.cat([x.view(-1) for x in model_2.linear4.parameters()][:-1])
    

            model_3_all_linear1_params = model_3.linear1
            model_3_all_linear2_params= torch.cat([x.view(-1) for x in model_3.linear2.parameters()][:-1])
            model_3_all_linear3_params= torch.cat([x.view(-1) for x in model_3.linear3.parameters()][:-1])
            model_3_all_linear4_params= torch.cat([x.view(-1) for x in model_3.linear4.parameters()][:-1])
    


            # compute loss
            all_linear1_params=torch.cat(( model_1_all_linear1_params.reshape(1,-1),  model_2_all_linear1_params.reshape(1,-1), model_3_all_linear1_params.reshape(1,-1)),0)
            all_linear234_params=torch.cat((model_1_all_linear2_params,model_1_all_linear3_params, model_1_all_linear4_params,
                                             model_2_all_linear2_params,model_2_all_linear3_params, model_2_all_linear4_params,
                                             model_3_all_linear2_params,model_3_all_linear3_params,model_3_all_linear4_params),0)

            non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)

            BCE_loss_1 = loss_fn(out_1, targets_1[idx_train].reshape(-1,1))
            BCE_loss_2 = loss_fn(out_2, targets_2[idx_train].reshape(-1,1))
            BCE_loss_3 = loss_fn(out_3, targets_3[idx_train].reshape(-1,1))

#             l1_regularization = lambda1 * smooth_l1(all_linear1_params,prior)
            l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

            l2_regularization = lambda2* torch.sum(all_linear234_params.pow(2))
            BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
            loss =BCE_loss+l1_regularization +l2_regularization

            # record loss
            BCE.append(BCE_loss.item())

            # compute derivative
            loss.backward()

            # gradient descent
            optimizer.step()

            # learning rate decay
            scheduler.step()
        
        
        test_out_1, test_layer1_out_1, test_layer2_out_1,test_layer3_out_1= model_1(inputs_1[idx_test])
        test_out_2, test_layer1_out_2, test_layer2_out_2,test_layer3_out_2= model_2(inputs_2[idx_test])
        test_out_3, test_layer1_out_3, test_layer2_out_3,test_layer3_out_3= model_3(inputs_3[idx_test])
        

        test_prediction_1= (model_1(inputs_1[idx_test])[0]>0.5).clone().int()
        test_target_1= train_y1[idx_test].reshape(-1,1).int()

        test_prediction_2= (model_2(inputs_2[idx_test])[0]>0.5).clone().int()
        test_target_2= train_y2[idx_test].reshape(-1,1).int()

        test_prediction_3= (model_3(inputs_3[idx_test])[0]>0.5).clone().int()
        test_target_3= train_y3[idx_test].reshape(-1,1).int()

        test_prediction=np.append(test_prediction_1,test_prediction_2)
        test_prediction=np.append(test_prediction,test_prediction_3)
        test_prediction=test_prediction.tolist()

        test_target=np.append(test_target_1,test_target_2)
        test_target=np.append(test_target,test_target_3)
        test_target=test_target.tolist()

        final_matrix=confusion_matrix(test_target,test_prediction)
        tTPR=recall_score(test_target,test_prediction)#TPR
        tTNR=final_matrix[0,0]/(sum(final_matrix[0,:]))#TNR
        tGM=math.sqrt(tTPR*tTNR)

        sum_tGM+=tGM

    tune.report(my_test_sum_tGM=sum_tGM/split_num)


# In[ ]:


def main_final_model(lambda1,lambda2,lr,eta,ga,inputs_1,inputs_2,inputs_3,train_y1,train_y2,train_y3,y_prior_1,y_prior_2,y_prior_3,
                     test_inputs_1,test_inputs_2,test_inputs_3,test_targets_1,test_targets_2,test_targets_3,prior):
    learning_rate=lr
    max_iteration=1000

    # record loss descent
    BCE=[]
    targets_1=(1-eta)*train_y1+eta*y_prior_1
    targets_2=(1-eta)*train_y2+eta*y_prior_2
    targets_3=(1-eta)*train_y3+eta*y_prior_3

    # main nn object

    model_1=MLP_final(seed=1)
    model_2=MLP_final(seed=1)
    model_3=MLP_final(seed=1)

    loss_fn = torch.nn.BCELoss()
    params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
    optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)

    # learning rate decay scheme
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

    loss_record=[]
    # loop for max_iteration times
    for t in range(max_iteration):

        # renew optimizer
        optimizer.zero_grad()

        # forward propagate
        out_1, layer1_out_1, layer2_out_1,layer3_out_1= model_1(inputs_1)
        out_2, layer1_out_2, layer2_out_2,layer3_out_2= model_2(inputs_2)
        out_3, layer1_out_3, layer2_out_3,layer3_out_3= model_3(inputs_3)



        # extract parameters
        #[:-1] for leaving out bias term#
        model_1_all_linear1_params = model_1.linear1
        model_1_all_linear2_params= torch.cat([x.view(-1) for x in model_1.linear2.parameters()][:-1])
        model_1_all_linear3_params= torch.cat([x.view(-1) for x in model_1.linear3.parameters()][:-1])
        model_1_all_linear4_params= torch.cat([x.view(-1) for x in model_1.linear4.parameters()][:-1])

        model_2_all_linear1_params = model_1.linear1
        model_2_all_linear2_params= torch.cat([x.view(-1) for x in model_2.linear2.parameters()][:-1])
        model_2_all_linear3_params= torch.cat([x.view(-1) for x in model_2.linear3.parameters()][:-1])
        model_2_all_linear4_params= torch.cat([x.view(-1) for x in model_2.linear4.parameters()][:-1])

        model_3_all_linear1_params = model_3.linear1
        model_3_all_linear2_params= torch.cat([x.view(-1) for x in model_3.linear2.parameters()][:-1])
        model_3_all_linear3_params= torch.cat([x.view(-1) for x in model_3.linear3.parameters()][:-1])
        model_3_all_linear4_params= torch.cat([x.view(-1) for x in model_3.linear4.parameters()][:-1])


        # compute loss

        all_linear1_params=torch.cat(( model_1_all_linear1_params.reshape(1,-1),  model_2_all_linear1_params.reshape(1,-1), model_3_all_linear1_params.reshape(1,-1)),0)
        all_linear234_params=torch.cat((model_1_all_linear2_params,model_1_all_linear3_params, model_1_all_linear4_params,
                                         model_2_all_linear2_params,model_2_all_linear3_params, model_2_all_linear4_params,
                                         model_3_all_linear2_params,model_3_all_linear3_params,model_3_all_linear4_params),0)

        BCE_loss_1 = loss_fn(out_1, targets_1.reshape(-1,1))
        BCE_loss_2 = loss_fn(out_2, targets_2.reshape(-1,1))
        BCE_loss_3 = loss_fn(out_3, targets_3.reshape(-1,1))

        non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)

        l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))
        l2_regularization = lambda2 * torch.sum(all_linear234_params.pow(2))
        BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
        loss =BCE_loss+l1_regularization +l2_regularization

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
    model_2_weight=model_1_all_linear1_params.detach().numpy().copy().reshape(1,-1)
    model_3_weight=model_1_all_linear1_params.detach().numpy().copy().reshape(1,-1)
    
    return prediction,target,model_1_weight,model_2_weight,model_3_weight

# ## M2

# In[ ]:


def train_model_no_prior_final(config):
    split_num=5
    kf = KFold(n_splits=split_num)
    
    
    lambda1,lambda2,lr,ga,= config["lambda1"], config["lambda2"],config["lr"],config["ga"]
    inputs_1,targets_1 = config["inputs_1"],config["targets_1"]
    inputs_2,targets_2 = config["inputs_2"],config["targets_2"]
    inputs_3,targets_3 = config["inputs_3"],config["targets_3"]
    
    prior=config["prior"]
        
    sum_tGM=0
    for idx_train,idx_test in kf.split(inputs_1):
       
        
        model_1=MLP_final(seed=1)
        model_2=MLP_final(seed=2)
        model_3=MLP_final(seed=1)
        max_iteration=1000
        learning_rate=lr
        params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
        optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
        loss_fn = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

        BCE=[]

        for t in range(max_iteration):
            optimizer.zero_grad() # renew optimizer
            out_1, layer1_out_1, layer2_out_1,layer3_out_1= model_1(inputs_1[idx_train])
            out_2, layer1_out_2, layer2_out_2,layer3_out_2= model_2(inputs_2[idx_train])
            out_3, layer1_out_3, layer2_out_3,layer3_out_3= model_3(inputs_3[idx_train])# forward propagate

            # extract parameters
            #[:-1] for leaving out bias term#
            model_1_all_linear1_params = model_1.linear1
            model_1_all_linear2_params= torch.cat([x.view(-1) for x in model_1.linear2.parameters()][:-1])
            model_1_all_linear3_params= torch.cat([x.view(-1) for x in model_1.linear3.parameters()][:-1])
            model_1_all_linear4_params= torch.cat([x.view(-1) for x in model_1.linear4.parameters()][:-1])
           

            model_2_all_linear1_params = model_1.linear1
            model_2_all_linear2_params= torch.cat([x.view(-1) for x in model_2.linear2.parameters()][:-1])
            model_2_all_linear3_params= torch.cat([x.view(-1) for x in model_2.linear3.parameters()][:-1])
            model_2_all_linear4_params= torch.cat([x.view(-1) for x in model_2.linear4.parameters()][:-1])
    

            model_3_all_linear1_params = model_3.linear1
            model_3_all_linear2_params= torch.cat([x.view(-1) for x in model_3.linear2.parameters()][:-1])
            model_3_all_linear3_params= torch.cat([x.view(-1) for x in model_3.linear3.parameters()][:-1])
            model_3_all_linear4_params= torch.cat([x.view(-1) for x in model_3.linear4.parameters()][:-1])
    


            # compute loss
            all_linear1_params=torch.cat(( model_1_all_linear1_params.reshape(1,-1),  model_2_all_linear1_params.reshape(1,-1), model_3_all_linear1_params.reshape(1,-1)),0)
            all_linear234_params=torch.cat((model_1_all_linear2_params,model_1_all_linear3_params, model_1_all_linear4_params,
                                             model_2_all_linear2_params,model_2_all_linear3_params, model_2_all_linear4_params,
                                             model_3_all_linear2_params,model_3_all_linear3_params,model_3_all_linear4_params),0)

            non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)

            BCE_loss_1 = loss_fn(out_1, targets_1[idx_train].reshape(-1,1))
            BCE_loss_2 = loss_fn(out_2, targets_2[idx_train].reshape(-1,1))
            BCE_loss_3 = loss_fn(out_3, targets_3[idx_train].reshape(-1,1))

#             l1_regularization = lambda1 * smooth_l1(all_linear1_params,prior)
            l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

            l2_regularization = lambda2* torch.sum(all_linear234_params.pow(2))
            BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
            loss =BCE_loss+l1_regularization +l2_regularization

            # record loss
            BCE.append(BCE_loss.item())

            # compute derivative
            loss.backward()

            # gradient descent
            optimizer.step()

            # learning rate decay
            scheduler.step()
        
        
        test_out_1, test_layer1_out_1, test_layer2_out_1,test_layer3_out_1= model_1(inputs_1[idx_test])
        test_out_2, test_layer1_out_2, test_layer2_out_2,test_layer3_out_2= model_2(inputs_2[idx_test])
        test_out_3, test_layer1_out_3, test_layer2_out_3,test_layer3_out_3= model_3(inputs_3[idx_test])
        

        test_prediction_1= (model_1(inputs_1[idx_test])[0]>0.5).clone().int()
        test_target_1= targets_1[idx_test].reshape(-1,1).int()

        test_prediction_2= (model_2(inputs_2[idx_test])[0]>0.5).clone().int()
        test_target_2= targets_2[idx_test].reshape(-1,1).int()

        test_prediction_3= (model_3(inputs_3[idx_test])[0]>0.5).clone().int()
        test_target_3= targets_3[idx_test].reshape(-1,1).int()

        test_prediction=np.append(test_prediction_1,test_prediction_2)
        test_prediction=np.append(test_prediction,test_prediction_3)
        test_prediction=test_prediction.tolist()

        test_target=np.append(test_target_1,test_target_2)
        test_target=np.append(test_target,test_target_3)
        test_target=test_target.tolist()

        final_matrix=confusion_matrix(test_target,test_prediction)
        tTPR=recall_score(test_target,test_prediction)#TPR
        tTNR=final_matrix[0,0]/(sum(final_matrix[0,:]))#TNR
        tGM=math.sqrt(tTPR*tTNR)

        sum_tGM+=tGM

    tune.report(my_test_sum_tGM=sum_tGM/split_num)

    
    
# In[ ]:


def no_prior_final_model(lambda1,lambda2,lr,ga,inputs_1,inputs_2,inputs_3,targets_1,targets_2,targets_3,
                     test_inputs_1,test_inputs_2,test_inputs_3,test_targets_1,test_targets_2,test_targets_3,prior):
    learning_rate=lr
    max_iteration=1000

    # record loss descent
    BCE=[]
    
    # main nn object

    model_1=MLP_final(seed=1)
    model_2=MLP_final(seed=1)
    model_3=MLP_final(seed=1)

    loss_fn = torch.nn.BCELoss()
    params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
    optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)

    # learning rate decay scheme
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

    loss_record=[]
    # loop for max_iteration times
    for t in range(max_iteration):

        # renew optimizer
        optimizer.zero_grad()

        # forward propagate
        out_1, layer1_out_1, layer2_out_1,layer3_out_1= model_1(inputs_1)
        out_2, layer1_out_2, layer2_out_2,layer3_out_2= model_2(inputs_2)
        out_3, layer1_out_3, layer2_out_3,layer3_out_3= model_3(inputs_3)



        # extract parameters
        #[:-1] for leaving out bias term#
        model_1_all_linear1_params = model_1.linear1
        model_1_all_linear2_params= torch.cat([x.view(-1) for x in model_1.linear2.parameters()][:-1])
        model_1_all_linear3_params= torch.cat([x.view(-1) for x in model_1.linear3.parameters()][:-1])
        model_1_all_linear4_params= torch.cat([x.view(-1) for x in model_1.linear4.parameters()][:-1])

        model_2_all_linear1_params = model_1.linear1
        model_2_all_linear2_params= torch.cat([x.view(-1) for x in model_2.linear2.parameters()][:-1])
        model_2_all_linear3_params= torch.cat([x.view(-1) for x in model_2.linear3.parameters()][:-1])
        model_2_all_linear4_params= torch.cat([x.view(-1) for x in model_2.linear4.parameters()][:-1])

        model_3_all_linear1_params = model_3.linear1
        model_3_all_linear2_params= torch.cat([x.view(-1) for x in model_3.linear2.parameters()][:-1])
        model_3_all_linear3_params= torch.cat([x.view(-1) for x in model_3.linear3.parameters()][:-1])
        model_3_all_linear4_params= torch.cat([x.view(-1) for x in model_3.linear4.parameters()][:-1])


        # compute loss

        all_linear1_params=torch.cat(( model_1_all_linear1_params.reshape(1,-1),  model_2_all_linear1_params.reshape(1,-1), model_3_all_linear1_params.reshape(1,-1)),0)
        all_linear234_params=torch.cat((model_1_all_linear2_params,model_1_all_linear3_params, model_1_all_linear4_params,
                                         model_2_all_linear2_params,model_2_all_linear3_params, model_2_all_linear4_params,
                                         model_3_all_linear2_params,model_3_all_linear3_params,model_3_all_linear4_params),0)

        BCE_loss_1 = loss_fn(out_1, targets_1.reshape(-1,1))
        BCE_loss_2 = loss_fn(out_2, targets_2.reshape(-1,1))
        BCE_loss_3 = loss_fn(out_3, targets_3.reshape(-1,1))

        non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)

        l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))
        l2_regularization = lambda2 * torch.sum(all_linear234_params.pow(2))
        BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
        loss =BCE_loss+l1_regularization +l2_regularization

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
    model_2_weight=model_1_all_linear1_params.detach().numpy().copy().reshape(1,-1)
    model_3_weight=model_1_all_linear1_params.detach().numpy().copy().reshape(1,-1)
    
    return prediction,target,model_1_weight,model_2_weight,model_3_weight


    
# In[ ]:


def train_model_pDFS_prior(config):
    split_num=5
    kf = KFold(n_splits=split_num)

    inputs,targets = config["inputs"],config["targets"]
    
    prior=config["prior"]

    sum_test_BCE_loss=0
    for idx_train,idx_test in kf.split(inputs):
        lambda1,lambda2,lr,ga= config["lambda1"], config["lambda2"],config["lr"],config["ga"]
        
        model=MLP_prior(seed=1)
        max_iteration=1000
        learning_rate=lr
        params_to_optimize=list(model.parameters())
        optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
        loss_fn = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

        BCE=[]

        for t in range(max_iteration):
            optimizer.zero_grad() # renew optimizer
            out, layer1_out, layer2_out= model(inputs[idx_train])

            # extract parameters
            #[:-1] for leaving out bias term#
            model_all_linear1_params = model.linear1
            model_all_linear2_params= torch.cat([x.view(-1) for x in model.linear2.parameters()][:-1])
            model_all_linear3_params= torch.cat([x.view(-1) for x in model.linear3.parameters()][:-1])

    


            # compute loss
            all_linear1_params=model_all_linear1_params.reshape(1,-1)
            all_linear23_params=torch.cat((model_all_linear2_params,model_all_linear3_params),0)
            
            non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)
            BCE_loss = loss_fn(out, targets[idx_train].reshape(-1,1))


#             l1_regularization = lambda1 * smooth_l1(all_linear1_params,prior)
            l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

            l2_regularization = lambda2* torch.sum(all_linear23_params.pow(2))

            loss =BCE_loss+l1_regularization +l2_regularization

            # record loss
            BCE.append(BCE_loss.item())

            # compute derivative
            loss.backward()

            # gradient descent
            optimizer.step()

            # learning rate decay
            scheduler.step()
        
        
        test_out, test_layer1_out, test_layer2_out= model(inputs[idx_test])

        
        test_BCE_loss = loss_fn(test_out, targets[idx_test].reshape(-1,1))

        
        sum_test_BCE_loss+=test_BCE_loss.item()

    #print("loss: ",)
    tune.report(my_test_BCE_loss=sum_test_BCE_loss/split_num)
    
    
# In[ ]:


def pDFS_prior_model(lambda1,lambda2,lr,ga,inputs,targets,prior):   
    learning_rate=lr
    max_iteration=1000
    
    BCE=[]
    
    model=MLP_prior(seed=1)

    loss_fn = torch.nn.BCELoss()
    params_to_optimize=list(model.parameters())
    optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

    loss_record=[]
    
    for t in range(max_iteration):
    
        # renew optimizer
        optimizer.zero_grad()

        # forward propagate
        out, layer1_out, layer2_out= model(inputs)



        # extract parameters
        #[:-1] for leaving out bias term#
        model_all_linear1_params = model.linear1
        model_all_linear2_params= torch.cat([x.view(-1) for x in model.linear2.parameters()][:-1])
        model_all_linear3_params= torch.cat([x.view(-1) for x in model.linear3.parameters()][:-1])

        # compute loss

        all_linear1_params=model_all_linear1_params.reshape(1,-1)
        all_linear23_params=torch.cat((model_all_linear2_params,model_all_linear3_params),0)

        BCE_loss = loss_fn(out, targets.reshape(-1,1))
        
        non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)
    #     l1_regularization = lambda1 * smooth_l1(all_linear1_params,prior)

        l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

        l2_regularization = lambda2 * torch.sum(all_linear23_params.pow(2))

        loss =BCE_loss+l1_regularization +l2_regularization

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
    y_prior= (model(inputs)[0]).detach().squeeze()
    return y_prior

# In[ ]:



def train_model_pDFS_final(config):
    split_num=5
    kf = KFold(n_splits=split_num)
    
    
    
        
    lambda1,lambda2,eta,lr,ga,= config["lambda1"], config["lambda2"],config["eta"],config["lr"],config["ga"]
    inputs,train_y,y_prior= config["inputs"],config["train_y"],config["y_prior"]
    
    prior=config["prior"]
    
    targets=(1-eta)*train_y+eta*y_prior
    
    
    
    sum_tGM=0
    for idx_train,idx_test in kf.split(inputs):
       
        
        model=MLP_final(seed=1)

        max_iteration=1000
        learning_rate=lr
        params_to_optimize=list(model.parameters())
        optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
        loss_fn = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

        BCE=[]

        for t in range(max_iteration):
            optimizer.zero_grad() # renew optimizer
            out, layer1_out, layer2_out,layer3_out= model(inputs[idx_train])


            # extract parameters
            #[:-1] for leaving out bias term#
            model_all_linear1_params = model.linear1
            model_all_linear2_params= torch.cat([x.view(-1) for x in model.linear2.parameters()][:-1])
            model_all_linear3_params= torch.cat([x.view(-1) for x in model.linear3.parameters()][:-1])
            model_all_linear4_params= torch.cat([x.view(-1) for x in model.linear4.parameters()][:-1])
    


            # compute loss
            all_linear1_params=model_all_linear1_params.reshape(1,-1)
            all_linear234_params=torch.cat((model_all_linear2_params,model_all_linear3_params, model_all_linear4_params),0)

            BCE_loss = loss_fn(out, targets[idx_train].reshape(-1,1))
            
            non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)
#             l1_regularization = lambda1 * smooth_l1(all_linear1_params,prior)
            l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

            l2_regularization = lambda2* torch.sum(all_linear234_params.pow(2))
            loss =BCE_loss+l1_regularization +l2_regularization

            # record loss
            BCE.append(BCE_loss.item())

            # compute derivative
            loss.backward()

            # gradient descent
            optimizer.step()

            # learning rate decay
            scheduler.step()
        
        
        test_out, test_layer1_out, test_layer2_out,test_layer3_out= model(inputs[idx_test])

        test_prediction= (model(inputs[idx_test])[0]>0.5).clone().int()
        test_target= train_y[idx_test].reshape(-1,1).int()
        
        test_prediction=test_prediction.tolist()

        test_target=test_target.tolist()

        final_matrix=confusion_matrix(test_target,test_prediction)
        tTPR=recall_score(test_target,test_prediction)#TPR
        tTNR=final_matrix[0,0]/(sum(final_matrix[0,:]))#TNR
        tGM=math.sqrt(tTPR*tTNR)

        sum_tGM+=tGM

    tune.report(my_test_sum_tGM=sum_tGM/split_num)
    
    
# In[ ]:



def pDFS_final_model(lambda1,lambda2,lr,ga,eta,inputs,train_y,y_prior,test_inputs,test_targets,prior):   
    
   
    learning_rate=lr
    max_iteration=1000

    targets=(1-eta)*train_y+eta*y_prior

    # record loss descent
    BCE=[]



    model=MLP_final(seed=1)

    loss_fn = torch.nn.BCELoss()
    params_to_optimize=list(model.parameters())
    optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)

    # learning rate decay scheme
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

    loss_record=[]
    # loop for max_iteration times
    for t in range(max_iteration):

        # renew optimizer
        optimizer.zero_grad()

        # forward propagate
        out, layer1_out, layer2_out,layer3_out= model(inputs)



        model_all_linear1_params = model.linear1
        model_all_linear2_params= torch.cat([x.view(-1) for x in model.linear2.parameters()][:-1])
        model_all_linear3_params= torch.cat([x.view(-1) for x in model.linear3.parameters()][:-1])
        model_all_linear4_params= torch.cat([x.view(-1) for x in model.linear4.parameters()][:-1])

        # compute loss

        all_linear1_params=model_all_linear1_params.reshape(1,-1)
        all_linear234_params=torch.cat((model_all_linear2_params,model_all_linear3_params, model_all_linear4_params),0)

        BCE_loss = loss_fn(out, targets.reshape(-1,1))

        non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)

        l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))
        l2_regularization = lambda2 * torch.sum(all_linear234_params.pow(2))
        loss =BCE_loss+l1_regularization +l2_regularization

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

    prediction= (model(test_inputs)[0]>0.5).clone().int()
    target= test_targets.reshape(-1,1).int()

    prediction=prediction.tolist()
    target=target.tolist()
    
    model_weight=model_all_linear1_params.detach().numpy().copy().reshape(1,-1)
    
    return prediction,target,model_weight


# In[ ]:




def train_model_IA_prior(config):
    
     # switch training set
    inputs_1,targets_1 = config["inputs_1"],config["targets_1"]
    inputs_2,targets_2 = config["inputs_2"],config["targets_2"]
    inputs_3,targets_3 = config["inputs_3"],config["targets_3"]
    
    prior=config["prior"]

    split_num=5
    kf = KFold(n_splits=split_num)

    sum_test_BCE_loss=0
    for idx_train,idx_test in kf.split(inputs_1):
        lambda1,lr,ga= config["lambda1"], config["lr"],config["ga"]
        
        
        model_1=MLP_inte(seed=1)
        model_2=MLP_inte(seed=1)
        model_3=MLP_inte(seed=1)
        
        
        max_iteration=1000
        learning_rate=lr
        params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
        optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
        loss_fn = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

        BCE=[]

        for t in range(max_iteration):
            optimizer.zero_grad() # renew optimizer
            out_1= model_1(inputs_1[idx_train])
            out_2= model_2(inputs_2[idx_train])
            out_3= model_3(inputs_3[idx_train])# forward propagate

            # extract parameters
            #[:-1] for leaving out bias term#
            model_1_all_linear1_params= torch.cat([x.view(-1) for x in model_1.linear1.parameters()][:-1])
           

            model_2_all_linear1_params= torch.cat([x.view(-1) for x in model_2.linear1.parameters()][:-1])
    

            model_3_all_linear1_params= torch.cat([x.view(-1) for x in model_3.linear1.parameters()][:-1])
    


            # compute loss
            all_linear1_params=torch.cat(( model_1_all_linear1_params.reshape(1,-1),  model_2_all_linear1_params.reshape(1,-1), model_3_all_linear1_params.reshape(1,-1)),0)
            
            non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)
            
            BCE_loss_1 = loss_fn(out_1, targets_1[idx_train].reshape(-1,1))
            BCE_loss_2 = loss_fn(out_2, targets_2[idx_train].reshape(-1,1))
            BCE_loss_3 = loss_fn(out_3, targets_3[idx_train].reshape(-1,1))

#             l1_regularization = lambda1 * smooth_l1(all_linear1_params,prior)
            l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

#             l2_regularization = lambda2* torch.sum(all_linear23_params.pow(2))
            BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
            loss =BCE_loss+l1_regularization

            # record loss
            BCE.append(BCE_loss.item())

            # compute derivative
            loss.backward()

            # gradient descent
            optimizer.step()

            # learning rate decay
            scheduler.step()
        
        
        test_out_1= model_1(inputs_1[idx_test])
        test_out_2= model_2(inputs_2[idx_test])
        test_out_3= model_3(inputs_3[idx_test])
        
        test_BCE_loss_1 = loss_fn(test_out_1, targets_1[idx_test].reshape(-1,1))
        test_BCE_loss_2 = loss_fn(test_out_2, targets_2[idx_test].reshape(-1,1))
        test_BCE_loss_3 = loss_fn(test_out_3, targets_3[idx_test].reshape(-1,1))
        
        test_BCE_loss=test_BCE_loss_1+test_BCE_loss_2+test_BCE_loss_3
        
        sum_test_BCE_loss+=test_BCE_loss.item()

    #print("loss: ",)
    tune.report(my_test_BCE_loss=sum_test_BCE_loss/split_num)
    
    
    
# In[ ]:

def IA_prior_model(lambda1,lr,ga,inputs_1,inputs_2,inputs_3,targets_1,targets_2,targets_3,prior):
    learning_rate=lr
    max_iteration=1000
    
    BCE=[]
    
    model_1=MLP_inte(seed=1)
    model_2=MLP_inte(seed=2)
    model_3=MLP_inte(seed=1)

    loss_fn = torch.nn.BCELoss()
    params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
    optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

    loss_record=[]
    
    for t in range(max_iteration):
    
        # renew optimizer
        optimizer.zero_grad()

        # forward propagate
        out_1 = model_1(inputs_1)
        out_2 = model_2(inputs_2)
        out_3 = model_3(inputs_3)



        # extract parameters

        model_1_all_linear1_params= torch.cat([x.view(-1) for x in model_1.linear1.parameters()][:-1])


        model_2_all_linear1_params= torch.cat([x.view(-1) for x in model_2.linear1.parameters()][:-1])


        model_3_all_linear1_params= torch.cat([x.view(-1) for x in model_3.linear1.parameters()][:-1])


        # compute loss

        all_linear1_params=torch.cat(( model_1_all_linear1_params.reshape(1,-1),  model_2_all_linear1_params.reshape(1,-1), model_3_all_linear1_params.reshape(1,-1)),0)

        BCE_loss_1 = loss_fn(out_1, targets_1.reshape(-1,1))
        BCE_loss_2 = loss_fn(out_2, targets_2.reshape(-1,1))
        BCE_loss_3 = loss_fn(out_3, targets_3.reshape(-1,1))
        
        non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)
    #     l1_regularization = lambda1 * smooth_l1(all_linear1_params,prior)

        l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

        BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
        
        loss =BCE_loss+l1_regularization
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
    y_prior_1= (model_1(inputs_1)).detach().squeeze()

    y_prior_2= (model_2(inputs_2)).detach().squeeze()

    y_prior_3= (model_3(inputs_3)).detach().squeeze()
    
    return y_prior_1,y_prior_2,y_prior_3



# In[ ]:



def train_model_IA_final(config):
    split_num=5
    kf = KFold(n_splits=split_num)
    
    
    lambda1,eta,lr,ga,= config["lambda1"], config["eta"],config["lr"],config["ga"]
    inputs_1,train_y1 = config["inputs_1"],config["train_y1"]
    inputs_2,train_y2 = config["inputs_2"],config["train_y2"]
    inputs_3,train_y3 = config["inputs_3"],config["train_y3"]
    
    prior=config["prior"]
    
    y_prior_1=config["y_prior_1"]
    y_prior_2=config["y_prior_2"]
    y_prior_3=config["y_prior_3"]
    
    targets_1=(1-eta)*train_y1+eta*y_prior_1
    targets_2=(1-eta)*train_y2+eta*y_prior_2
    targets_3=(1-eta)*train_y3+eta*y_prior_3
    
    
    sum_tGM=0
    for idx_train,idx_test in kf.split(inputs_1):
       
        
        model_1=MLP_inte(seed=1)
        model_2=MLP_inte(seed=2)
        model_3=MLP_inte(seed=1)
        
        max_iteration=1000
        learning_rate=lr
        params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
        optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)
        loss_fn = torch.nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

        BCE=[]

        for t in range(max_iteration):
            optimizer.zero_grad() # renew optimizer
            out_1 = model_1(inputs_1[idx_train])
            out_2 = model_2(inputs_2[idx_train])
            out_3 = model_3(inputs_3[idx_train])# forward propagate

            # extract parameters
            #[:-1] for leaving out bias term#
            model_1_all_linear1_params= torch.cat([x.view(-1) for x in model_1.linear1.parameters()][:-1])
           

            model_2_all_linear1_params= torch.cat([x.view(-1) for x in model_2.linear1.parameters()][:-1])
    

            model_3_all_linear1_params= torch.cat([x.view(-1) for x in model_3.linear1.parameters()][:-1])
    


            # compute loss
            all_linear1_params=torch.cat(( model_1_all_linear1_params.reshape(1,-1),  model_2_all_linear1_params.reshape(1,-1), model_3_all_linear1_params.reshape(1,-1)),0)

            non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)

            BCE_loss_1 = loss_fn(out_1, targets_1[idx_train].reshape(-1,1))
            BCE_loss_2 = loss_fn(out_2, targets_2[idx_train].reshape(-1,1))
            BCE_loss_3 = loss_fn(out_3, targets_3[idx_train].reshape(-1,1))

#             l1_regularization = lambda1 * smooth_l1(all_linear1_params,prior)
            l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

            BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
            loss =BCE_loss+l1_regularization

            # record loss
            BCE.append(BCE_loss.item())

            # compute derivative
            loss.backward()

            # gradient descent
            optimizer.step()

            # learning rate decay
            scheduler.step()
        
        
        test_out_1= model_1(inputs_1[idx_test])
        test_out_2= model_2(inputs_2[idx_test])
        test_out_3= model_3(inputs_3[idx_test])
        

        test_prediction_1= (model_1(inputs_1[idx_test])>0.5).clone().int()
        test_target_1= train_y1[idx_test].reshape(-1,1).int()

        test_prediction_2= (model_2(inputs_2[idx_test])>0.5).clone().int()
        test_target_2= train_y2[idx_test].reshape(-1,1).int()

        test_prediction_3= (model_3(inputs_3[idx_test])>0.5).clone().int()
        test_target_3= train_y3[idx_test].reshape(-1,1).int()

        test_prediction=np.append(test_prediction_1,test_prediction_2)
        test_prediction=np.append(test_prediction,test_prediction_3)
        test_prediction=test_prediction.tolist()

        test_target=np.append(test_target_1,test_target_2)
        test_target=np.append(test_target,test_target_3)
        test_target=test_target.tolist()

        final_matrix=confusion_matrix(test_target,test_prediction)
        tTPR=recall_score(test_target,test_prediction)#TPR
        tTNR=final_matrix[0,0]/(sum(final_matrix[0,:]))#TNR
        tGM=math.sqrt(tTPR*tTNR)

        sum_tGM+=tGM

    tune.report(my_test_sum_tGM=sum_tGM/split_num)


# In[ ]:



    
def IA_final_model(lambda1,lr,eta,ga,inputs_1,inputs_2,inputs_3,train_y1,train_y2,train_y3,y_prior_1,y_prior_2,y_prior_3,
                     test_inputs_1,test_inputs_2,test_inputs_3,test_targets_1,test_targets_2,test_targets_3,prior):
    learning_rate=lr
    max_iteration=1000

    # record loss descent
    BCE=[]

    targets_1=(1-eta)*train_y1+eta*y_prior_1
    targets_2=(1-eta)*train_y2+eta*y_prior_2
    targets_3=(1-eta)*train_y3+eta*y_prior_3
    
    # main nn object

    model_1=MLP_inte(seed=1)
    model_2=MLP_inte(seed=1)
    model_3=MLP_inte(seed=1)

    loss_fn = torch.nn.BCELoss()
    params_to_optimize=list(model_1.parameters())+list(model_2.parameters())+list(model_3.parameters())
    optimizer = torch.optim.SGD(params_to_optimize, lr=learning_rate)

    # learning rate decay scheme
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400,500,600,700,800], gamma=ga)

    loss_record=[]
    # loop for max_iteration times
    for t in range(max_iteration):

        # renew optimizer
        optimizer.zero_grad()

        # forward propagate
        out_1 = model_1(inputs_1)
        out_2 = model_2(inputs_2)
        out_3 = model_3(inputs_3)



        # extract parameters
        #[:-1] for leaving out bias term#
        model_1_all_linear1_params= torch.cat([x.view(-1) for x in model_1.linear1.parameters()][:-1])


        model_2_all_linear1_params= torch.cat([x.view(-1) for x in model_2.linear1.parameters()][:-1])


        model_3_all_linear1_params= torch.cat([x.view(-1) for x in model_3.linear1.parameters()][:-1])


        # compute loss

        all_linear1_params=torch.cat(( model_1_all_linear1_params.reshape(1,-1),  model_2_all_linear1_params.reshape(1,-1), model_3_all_linear1_params.reshape(1,-1)),0)


        BCE_loss_1 = loss_fn(out_1, targets_1.reshape(-1,1))
        BCE_loss_2 = loss_fn(out_2, targets_2.reshape(-1,1))
        BCE_loss_3 = loss_fn(out_3, targets_3.reshape(-1,1))

        non_prior=np.setdiff1d([i for i in range(all_linear1_params.shape[1])],prior)

        l1_regularization = lambda1 * torch.sum(torch.norm(all_linear1_params[:,non_prior],p=2,dim=0))

        BCE_loss=BCE_loss_1+BCE_loss_2+BCE_loss_3
        loss =BCE_loss+l1_regularization

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
    prediction_1= (model_1(test_inputs_1)>0.5).clone().int()
    target_1= test_targets_1.reshape(-1,1).int()

    prediction_2= (model_2(test_inputs_2)>0.5).clone().int()
    target_2= test_targets_2.reshape(-1,1).int()

    prediction_3= (model_3(test_inputs_3)>0.5).clone().int()
    target_3= test_targets_3.reshape(-1,1).int()

    prediction=np.append(prediction_1,prediction_2)
    prediction=np.append(prediction,prediction_3)
    prediction=prediction.tolist()

    target=np.append(target_1,target_2)
    target=np.append(target,target_3)
    target=target.tolist()
    
    ####################################   variable selection ####################################     
    model_1_weight=model_1_all_linear1_params.detach().numpy().copy().reshape(1,-1)
    model_2_weight=model_1_all_linear1_params.detach().numpy().copy().reshape(1,-1)
    model_3_weight=model_1_all_linear1_params.detach().numpy().copy().reshape(1,-1)
    
    return prediction,target,model_1_weight,model_2_weight,model_3_weight