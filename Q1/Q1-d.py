#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import h5py
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
from sklearn.utils import shuffle


# In[2]:


def linear_predict(data_x,coef,intercept):
    pred_y = ((coef@data_x.T).T)+intercept
    for i in range(len(pred_y)):
        if (pred_y[i]<0.5):
            pred_y[i] = 0
        else:
            pred_y[i] = 1
    return pred_y
def accuracy(pred,given):
    s = 0
    for i in range(len(pred)):
        if (pred[i]==given[i]):
            s+=1
    return s*100/len(pred)
    


# In[3]:


def rbfKernel(x,y):
    ans = [0]*len(x)
    g = 0.7
    for i in range(len(x)):
        k=[0]*len(y)
        for j in range(len(y)):
            t = np.linalg.norm(x[i]-y[j])
            t = t**2
            t = -g*t
            t = np.exp(t) 
            k[j]=t
        ans[i] = k
    return ans
def get_sv(train,mod):
    sv = []
    for i in mod.support_:
        sv.append(train[i])
    return np.array(sv)
def pred(test,sv,mod):
    ans = []
    for i in range(len(test)):
        t = 0
        for j in range(len(sv)):
            t+=mod.dual_coef_[0][j]*np.exp(-0.7*np.linalg.norm(test[i]-sv[j])**2)
        t+=mod.intercept_
        ans.append(t)
    return ans
def rbfPredict(test,train,mod):
    sv = get_sv(train,mod)
    ans = pred(test,sv,mod)
    return ans
def comp(pred):
    for i in range(len(pred)):
        if (pred[i]<0):
            pred[i]=0
        else:
            pred[i]=1
    return pred
def accuracy_rbf(pred,given):
    s = 0
    pred = comp(pred)
    for i in range(len(pred)):
        if (pred[i]==given[i]):
            s+=1
    return s*100/len(pred)


# In[4]:


f1 = h5py.File('data_4.h5','r+') 
list(f1.keys())
X1 = f1['x']
y1=f1['y']
df1= np.array(X1[()])
dfy1= np.array(y1[()])

z = np.abs(stats.zscore(df1))
out = np.where(z>2)
cdf1 = []
cdfy1 = []
for i in range(len(df1)):
    if (i not in out[0]):
        cdf1.append(df1[i])
        cdfy1.append(dfy1[i])
df1 = np.array(cdf1)
dfy1= np.array(cdfy1)
df1,dfy1 = shuffle(df1,dfy1)
size = df1.shape[0]
train_x = df1[0:4*size//5]
train_y = dfy1[0:4*size//5]
val_x = df1[4*size//5:size]
val_y = dfy1[4*size//5:size]
mod = svm.SVC(kernel='linear')
mod.fit(train_x,train_y)
my_pred = linear_predict(val_x,mod.coef_,mod.intercept_)
print('---------Linear-data4-----------')
print('My accuracy for test =',accuracy(my_pred,val_y))

my_pred = linear_predict(train_x,mod.coef_,mod.intercept_)
print('My accuracy for train =',accuracy(my_pred,train_y))

t = mod.predict(val_x)
print('SVC accuracy for test =',100*mod.score(val_x,val_y))
t = mod.predict(train_x)
print('SVC accuracy for train =',100*mod.score(train_x,train_y))
print('-----------RBF-data4-----------')
mod = svm.SVC(kernel='rbf',gamma='scale')
mod.fit(train_x,train_y)
mod_self = svm.SVC(kernel=rbfKernel)
mod_self.fit(train_x,train_y)
print('SVC accuracy for test =',100*mod.score(val_x,val_y))
self_pred = rbfPredict(val_x,train_x,mod_self)
print('My accuracy for test =',accuracy_rbf(self_pred,val_y))

print('SVC accuracy for train =',100*mod.score(train_x,train_y))
self_pred = rbfPredict(train_x,train_x,mod_self)
print('My accuracy for train =',accuracy_rbf(self_pred,train_y))


# In[5]:


f1 = h5py.File('data_5.h5','r+') 
list(f1.keys())
X1 = f1['x']
y1=f1['y']
df1= np.array(X1[()])
dfy1= np.array(y1[()])
z = np.abs(stats.zscore(df1))
out = np.where(z>2)
cdf1 = []
cdfy1 = []
for i in range(len(df1)):
    if (i not in out[0]):
        cdf1.append(df1[i])
        cdfy1.append(dfy1[i])
df1 = np.array(cdf1)
dfy1= np.array(cdfy1)
df1,dfy1 = shuffle(df1,dfy1)
size = df1.shape[0]
train_x = df1[0:4*size//5]
train_y = dfy1[0:4*size//5]
val_x = df1[4*size//5:size]
val_y = dfy1[4*size//5:size]
mod = svm.SVC(kernel='linear')
mod.fit(train_x,train_y)
my_pred = linear_predict(val_x,mod.coef_,mod.intercept_)
print('---------Linear-data5-----------')
print('My accuracy for test =',accuracy(my_pred,val_y))

my_pred = linear_predict(train_x,mod.coef_,mod.intercept_)
print('My accuracy for train =',accuracy(my_pred,train_y))

t = mod.predict(val_x)
print('SVC accuracy for test =',100*mod.score(val_x,val_y))
t = mod.predict(train_x)
print('SVC accuracy for train =',100*mod.score(train_x,train_y))
print('-----------RBF-data5-----------')
mod = svm.SVC(kernel='rbf',gamma='scale')
mod.fit(train_x,train_y)
mod_self = svm.SVC(kernel=rbfKernel)
mod_self.fit(train_x,train_y)
print('SVC accuracy for test =',100*mod.score(val_x,val_y))
self_pred = rbfPredict(val_x,train_x,mod_self)
print('My accuracy for test =',accuracy_rbf(self_pred,val_y))

print('SVC accuracy for train =',100*mod.score(train_x,train_y))
self_pred = rbfPredict(train_x,train_x,mod_self)
print('My accuracy for train =',accuracy_rbf(self_pred,train_y))

