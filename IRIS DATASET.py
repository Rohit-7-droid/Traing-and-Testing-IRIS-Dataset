#!/usr/bin/env python
# coding: utf-8

# In[12]:


from sklearn.datasets import load_iris
import numpy as np
import keras
np.random.seed(10)


# In[13]:


iris = load_iris()
print('Keys in iris dictionary:',iris.keys())

X = iris['data']
print('First 3 entries of X:', '\n' , X[:3])

Y = iris['target']
print('First 3 entries of Y:', '\n' ,Y[:3])

names = iris['target_names']
print('names:' , names)
feature_names = iris['feature_names']
print('featurenames:' , feature_names)

#Track a few sample points 
isamples = np.random.randint(len(Y), size=(5))
print(isamples)


# In[14]:


#Shape of Data
print('shape of X :' , X.shape)
print('shape of Y :' , Y.shape)
print('X - samples: ', X[isamples])
print('Y - samples: ', Y[isamples])


# PRE PROCESS DATA
# Convert labels to categorical - One - hot encoding

# In[15]:


from keras.utils import to_categorical

Ny = len(np.unique(Y))  #Ny is number of categories/classes
print('Ny: ' , Ny)

Y = to_categorical(Y[:], num_classes = Ny )  # converted to 1 hot

print('X - samples:' , X[isamples])
print('Y - samples:' , Y[isamples])


# Train -Test Split
# 

# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# DATA NORMALIZING/SCALING
# Normalize data(X) to be of zero mean and unit variance

# In[17]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)  #compute the sum and standard deviation

X_train = scaler.transform(X_train) # perform transformation X= (X-mean)/std deviation
X_test = scaler.transform(X_test)
print('X_train: \n' , X_train[:5]) # FIRST 5 Samples
print('Y_train: \n' , Y_train[:5])


# TRAINING WITH AW=Y

# In[18]:


#define function to add column of 1's
addlcol = lambda x: np.concatenate((x,np.ones((x.shape[0],1))), axis =1)

Ns , Nx = X_train.shape
print('Ns: ', Ns, 'Nx: ' , Nx)

def find_weights(A,Y):
    print(A.shape)
    
    print(Y.shape)
    
    w = np.linalg.inv(A.T.dot(A)).dot(A.T.dot(Y))
    return w

A = addlcol(X_train) ##addlcol is a func that add a column of 1's
Y = Y_train
w = find_weights(A,Y) ## function find_weights gives pseudo inverse solution
print(w)


# In[20]:


def evaluate(X,W,Yd,transform_X_a):
    a = transform_X_a(X)
    yd = np.argmax(Yd , axis=1)
    y = np.argmax(a.dot(W),axis=1)
    print('confusion Matrix:')
    print(confusion_matrix(yd,y))
    
    
evaluate(X_train , w ,Y_train , addlcol) 
evaluate(X_test , w ,Y_test , addlcol) 


# GO WITH MODEL IN HIGHER DIMENSION BY INCREASING PARAMETERS BECAUSE 100% ACCURACY NOT OBTAINED

# In[21]:


addlcol = lambda x: np.concatenate((x,x**2,np.ones((x.shape[0],1))), axis =1)

A = addlcol(X_train) ##addlcol is a func that add square of each column and a  column of 1's
Y = Y_train
w = find_weights(A,Y) ## function find_weights gives pseudo inverse solution



evaluate(X_train , w ,Y_train , addlcol) 
evaluate(X_test , w ,Y_test , addlcol) 


# TRAIN ON 20 RANDOM SAMPLES

# In[26]:


addlcol = lambda x: np.concatenate((x,np.ones((x.shape[0],1))), axis =1)


## pick 12 random samples from datase 
ind = np.random.choice(range(X_train.shape[0]), size=12 , replace = False)
X_train_12 = X_train[ind]  ##X_train_12 contain 12 random samples from x_train
Y_train_12 = Y_train[ind]  ##Y_train_12 contain the corresponding Y values for the values in x_train


A = addlcol(X_train_12)
w = A.T.dot(np.linalg.inv(A.dot(A.T))).dot(Y_train_12) ##min norm sol an n<m

evaluate(X_train_12 , w ,Y_train_12 , addlcol) 
evaluate(X_test , w ,Y_test , addlcol) 


# In[ ]:




