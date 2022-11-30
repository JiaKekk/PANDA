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


# sigmoid
def sigmoid(X):
    return .5 * (1 + np.tanh(.5 * X))


# In[ ]:


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# In[ ]:


def cubic_root(x):
    return math.copysign(math.pow(abs(x), 1.0/3.0), x)


# In[ ]:


#linear
def generate_data_linear(corval,beta,n,p):
    mean=np.zeros(p)
    sigma=np.array([[corval**abs(i-j) for i in range(p)] for j in range(p)])
    x=torch.tensor(np.random.multivariate_normal(mean=mean,cov=sigma,size=n))
    Pi_test= sigmoid(x@beta)
    y=np.random.binomial(1,Pi_test.ravel(),n)
    x=x.to(torch.float32)
    #y=torch.tensor(y,dtype=torch.double)
    y=torch.tensor(y,dtype=torch.float)
    #y=y.to(torch.float32)
    return x,y


# In[ ]:


def sigmoid(X):
    return .5 * (1 + np.tanh(.5 * X))


# In[ ]:


#nolinear
def generate_data_nolinear(corval,beta,n,p,sin_index,exp_index):
    mean=np.zeros(p)
    sigma=np.array([[corval**abs(i-j) for i in range(p)] for j in range(p)])
    x_pre=torch.tensor(np.random.multivariate_normal(mean=mean,cov=sigma,size=n))
    x=torch.clone(x_pre)
    
    x[:,sin_index]=np.sin(4*np.pi*x_pre[:,sin_index])
    x[:,exp_index]=x_pre[:,exp_index]*x_pre[:,exp_index]*x_pre[:,exp_index]
    
    Pi_test= sigmoid(x@beta)
    y=np.random.binomial(1,Pi_test.ravel(),n)
    x=x.to(torch.float32)
    y=torch.tensor(y,dtype=torch.float)
    return x,y


# In[ ]:

