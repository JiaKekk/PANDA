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



# variable selection eveluation
def variable_selection_homo_DFS(threshold,model_weight,beta_1,p):
    mcl_w=np.zeros(p)

    for j in range(p):
        mcl_w[j]=model_weight[0,j]

    max_mcl_w=max(abs(mcl_w))
    for j in range(p):
        if (abs(mcl_w[j]))<=threshold*max_mcl_w:
            model_weight[0,j]=0
        else:
            model_weight[0,j]=model_weight[0,j]
    
    index=np.nonzero(model_weight)[1]
    
    beta=np.array(beta_1)
    trueindex=np.where(beta!=0)[0]
    trueindex=trueindex.tolist()
    trueindex=set(trueindex)
    
    index=np.nonzero(model_weight)[1]
    index=index.tolist()
    index=set(index)
    
    TP=len(trueindex.intersection(index))
    FP=len(np.setdiff1d(index,trueindex))
    FN=len(trueindex)-TP
    TN=p-len(trueindex)-FP
    
    return index,TP,FP,FN,TN
