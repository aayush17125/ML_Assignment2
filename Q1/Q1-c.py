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


# In[10]:


f1 = h5py.File('data_1.h5','r+') 
list(f1.keys())
X1 = f1['x']
y1=f1['y']
df1= np.array(X1[()])
dfy1= np.array(y1[()])
# print(df1,dfy1)
x0 = []
y0 = []
x1 = []
y1 = []
for i in range(len(df1)):
    if dfy1[i]==0:
        x0.append(df1[i][0])
        y0.append(df1[i][1])
    else:
        x1.append(df1[i][0])
        y1.append(df1[i][1])
z = np.abs(stats.zscore(df1))
out = np.where(z>2)
print(out)
cdf1 = []
cdfy1 = []
for i in range(len(df1)):
    if (i not in out[0]):
        cdf1.append(df1[i])
        cdfy1.append(dfy1[i])

cdf1 = np.array(cdf1)
cdfy1= np.array(cdfy1)
model1 = svm.SVC(kernel = 'poly',degree=2,gamma='scale')
model1.fit(cdf1,cdfy1)
plot_decision_regions(X=cdf1,y=cdfy1,clf=model1)
plt.show()

# In[11]:


f1 = h5py.File('data_2.h5','r+') 
list(f1.keys())
X1 = f1['x']
y1=f1['y']
df1= np.array(X1[()])
dfy1= np.array(y1[()])
# print(df1,dfy1)
x0 = []
y0 = []
x1 = []
y1 = []
for i in range(len(df1)):
    if dfy1[i]==0:
        x0.append(df1[i][0])
        y0.append(df1[i][1])
    else:
        x1.append(df1[i][0])
        y1.append(df1[i][1])
z = np.abs(stats.zscore(df1))
out = np.where(z>2)
print(out)
cdf1 = []
cdfy1 = []
for i in range(len(df1)):
    if (i not in out[0]):
        cdf1.append(df1[i])
        cdfy1.append(dfy1[i])
def kernel2(x,y):
    ans = ((x@y.T)+1)**3
    return ans
cdf1 = np.array(cdf1)
cdfy1= np.array(cdfy1)
model1 = svm.SVC(kernel = kernel2)
model1.fit(cdf1,cdfy1)
plot_decision_regions(X=cdf1,y=cdfy1,clf=model1)
plt.show()

# In[12]:


f1 = h5py.File('data_3.h5','r+') 
list(f1.keys())
X1 = f1['x']
y1=f1['y']
df1= np.array(X1[()])
dfy1= np.array(y1[()])
# print(df1,dfy1)
x0 = []
y0 = []
x1 = []
y1 = []
x2 = []
y2 = []
for i in range(len(df1)):
    if dfy1[i]==0:
        x0.append(df1[i][0])
        y0.append(df1[i][1])
    elif dfy1[i]==1:
        x1.append(df1[i][0])
        y1.append(df1[i][1])
    else:
        x2.append(df1[i][0])
        y2.append(df1[i][1])
z = np.abs(stats.zscore(df1))
out = np.where(z>2)
print(out)
cdf1 = []
cdfy1 = []
for i in range(len(df1)):
    if (i not in out[0]):
        cdf1.append(df1[i])
        cdfy1.append(dfy1[i])
cdf1 = np.array(cdf1)
cdfy1= np.array(cdfy1)
model1 = svm.SVC(kernel = 'linear')
model1.fit(cdf1,cdfy1)
plot_decision_regions(X=cdf1,y=cdfy1,clf=model1)
plt.show()

# In[5]:


f1 = h5py.File('data_4.h5','r+') 
list(f1.keys())
X1 = f1['x']
y1=f1['y']
df1= np.array(X1[()])
dfy1= np.array(y1[()])
# print(df1,dfy1)
x0 = []
y0 = []
x1 = []
y1 = []
for i in range(len(df1)):
    if dfy1[i]==0:
        x0.append(df1[i][0])
        y0.append(df1[i][1])
    else:
        x1.append(df1[i][0])
        y1.append(df1[i][1])
z = np.abs(stats.zscore(df1))
out = np.where(z>2)
print(out)
cdf1 = []
cdfy1 = []
for i in range(len(df1)):
    if (i not in out[0]):
        cdf1.append(df1[i])
        cdfy1.append(dfy1[i])
def kernel4(x,y):
    ans = ((x@y.T)+1)**2
    return ans
cdf1 = np.array(cdf1)
cdfy1= np.array(cdfy1)
model1 = svm.SVC(kernel = kernel4)
model1.fit(cdf1,cdfy1)
plot_decision_regions(X=cdf1,y=cdfy1,clf=model1)
plt.show()

# In[6]:


f1 = h5py.File('data_5.h5','r+') 
list(f1.keys())
X1 = f1['x']
y1=f1['y']
df1= np.array(X1[()])
dfy1= np.array(y1[()])
# print(df1,dfy1)
x0 = []
y0 = []
x1 = []
y1 = []
for i in range(len(df1)):
    if dfy1[i]==0:
        x0.append(df1[i][0])
        y0.append(df1[i][1])
    else:
        x1.append(df1[i][0])
        y1.append(df1[i][1])
z = np.abs(stats.zscore(df1))
out = np.where(z>2)
print(out)
cdf1 = []
cdfy1 = []
for i in range(len(df1)):
    if (i not in out[0]):
        cdf1.append(df1[i])
        cdfy1.append(dfy1[i])
def kernel5(x,y):
    ans = ((x@y.T)+1)**3
    return ans
cdf1 = np.array(cdf1)
cdfy1= np.array(cdfy1)
model1 = svm.SVC(kernel = kernel5)
model1.fit(cdf1,cdfy1)
plot_decision_regions(X=cdf1,y=cdfy1,clf=model1)
plt.show()

# In[ ]:




