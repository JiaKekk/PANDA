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
def cubic_root(x):
    return math.copysign(math.pow(abs(x), 1.0/3.0), x)


# variable selection eveluation
def variable_selection_homo_inte(threshold,model_1_weight,model_2_weight,model_3_weight,beta_1,p):
    mcl_w=np.zeros(p)

    for j in range(p):
        mcl_w[j]=cubic_root(model_1_weight[0,j]*model_2_weight[0,j]*model_3_weight[0,j])

    max_mcl_w=max(abs(mcl_w))
    
    for j in range(p):
        if (abs(mcl_w[j]))<=threshold*max_mcl_w:
            model_1_weight[0,j]=0
            model_2_weight[0,j]=0
            model_3_weight[0,j]=0
        else:
            model_1_weight[0,j]=model_1_weight[0,j]
            model_2_weight[0,j]=model_2_weight[0,j]
            model_3_weight[0,j]=model_3_weight[0,j]
    
    index=np.nonzero(model_1_weight)[1]

   
    beta=np.array(beta_1)
    trueindex=np.where(beta!=0)[0]
    trueindex=trueindex.tolist()
    trueindex=set(trueindex)
    index=np.nonzero(model_1_weight)[1]
    index=index.tolist()
    index=set(index)
    
    TP=len(trueindex.intersection(index))
    FP=len(index.difference(trueindex))
    FN=len(trueindex)-TP
    TN=p-len(trueindex)-FP
    
    return index,TP,FP,FN,TN


def variable_selection_hetero_inte(threshold,model_1_weight,model_2_weight,model_3_weight,beta_1,beta_2,beta_3,p):
    mcl_w_1=np.zeros(p)
    mcl_w_2=np.zeros(p)
    mcl_w_3=np.zeros(p)
    
    for i in range(p):
        mcl_w_1[i]=model_1_weight[0,i]
        mcl_w_2[i]=model_2_weight[0,i]
        mcl_w_3[i]=model_3_weight[0,i]

    max_mcl_w_1=max(abs(mcl_w_1))
    max_mcl_w_2=max(abs(mcl_w_2))
    max_mcl_w_3=max(abs(mcl_w_3))
    
    for i in range(p):
        if (abs(model_1_weight[0,i]))<=threshold*max_mcl_w_1:
            model_1_weight[0,i]=0
        else:
            model_1_weight[0,i]=model_1_weight[0,i]

    for i in range(p):
        if (abs(model_2_weight[0,i]))<=threshold*max_mcl_w_2:
            model_2_weight[0,i]=0
        else:
            model_2_weight[0,i]=model_2_weight[0,i]

    for i in range(p):
        if (abs(model_3_weight[0,i]))<=threshold*max_mcl_w_3:
            model_3_weight[0,i]=0
        else:
            model_3_weight[0,i]=model_3_weight[0,i]
    
    index_1=np.nonzero(model_1_weight)[1]
    index_2=np.nonzero(model_2_weight)[1]
    index_3=np.nonzero(model_3_weight)[1]
    
    index_1=index_1.tolist()
    index_1=set(index_1)
    beta_1=np.array(beta_1)
    trueindex_1=np.where(beta_1!=0)[0]
    trueindex_1=trueindex_1.tolist()
    trueindex_1=set(trueindex_1)

    index_2=np.nonzero(model_2_weight)[1]
    index_2=index_2.tolist()
    index_2=set(index_2)
    trueindex_2=np.where(beta_2!=0)[0]
    trueindex_2=trueindex_2.tolist()
    trueindex_2=set(trueindex_2)

    index_3=np.nonzero(model_3_weight)[1]
    index_3=index_3.tolist()
    index_3=set(index_3)
    trueindex_3=np.where(beta_3!=0)[0]
    trueindex_3=trueindex_3.tolist()
    trueindex_3=set(trueindex_3)


    TP_1=len(trueindex_1.intersection(index_1))
    FP_1=len(index_1.difference(trueindex_1))
    FN_1=len(trueindex_1)-TP_1
    TN_1=p-len(trueindex_1)-FP_1

    TP_2=len(trueindex_2.intersection(index_2))
    FP_2=len(index_2.difference(trueindex_2))
    FN_2=len(trueindex_2)-TP_2
    TN_2=p-len(trueindex_2)-FP_2

    TP_3=len(trueindex_3.intersection(index_3))
    FP_3=len(index_3.difference(trueindex_3))
    FN_3=len(trueindex_3)-TP_3
    TN_3=p-len(trueindex_3)-FP_3

   
    
    TP=TP_1+TP_2+TP_3
    FP=FP_1+FP_2+FP_3
    FN=FN_1+FN_2+FN_3
    TN=TN_1+TN_2+TN_3
    
    return TP,FP,FN,TN
