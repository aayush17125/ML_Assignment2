#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import h5py
import sys
import matplotlib.pyplot as plt


# In[2]:


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
plt.scatter(x0, y0, color= "blue", marker= "*", s=30)
plt.scatter(x1, y1, color= "red", marker= "+", s=30)
plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
# plot title 
plt.title('1')  
plt.show() 


# In[3]:


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
plt.scatter(x0, y0, color= "blue", marker= "*", s=30)
plt.scatter(x1, y1, color= "red", marker= "+", s=30)
plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
# plot title 
plt.title('1')  
plt.show() 


# In[4]:


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
plt.scatter(x0, y0, color= "blue", marker= "*", s=30)
plt.scatter(x1, y1, color= "red", marker= "+", s=30)
plt.scatter(x2, y2, color= "green", s=30)
plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
# plot title 
plt.title('1')  
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
plt.scatter(x0, y0, color= "blue", marker= "*", s=30)
plt.scatter(x1, y1, color= "red", marker= "+", s=30)
plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
# plot title 
plt.title('1')  
plt.show() 


# In[ ]:


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
plt.scatter(x0, y0, color= "blue", marker= "*", s=30)
plt.scatter(x1, y1, color= "red", marker= "+", s=30)
plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
# plot title 
plt.title('1')  
plt.show() 


# In[ ]:




