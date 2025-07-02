import numpy as np
import matplotlib as plt
import pandas as pd
from numpy.ma.core import shape
from pandas import DataFrame

path = r"C:\MATERIAL\CS 4\data2.txt"

data = pd.read_csv(path,names=['x1','x2','y1'] )
data=(data-data.mean())/data.std()
data.insert(0,'X0',1)
cols=data.shape[1]
m=data.shape[0]
x=data.iloc[:,0:cols-1]
y=data.iloc[:,cols-1:cols]

# convert to matrix

x=np.matrix(x.values)
y=np.matrix(y.values)
theta=np.matrix(np.zeros((1,cols-1)))

def compute_cost(x,y,theta):
    h = x * theta.T
    term= np.power(h-y,2)
    return 1/(2*m) *np.sum(term)


def gradient_descent(x,y,theta,alpha ,iter):
    parameters = int(theta.ravel().shape[1])
    cost=np.zeros(iter)
    tmp = np.matrix(np.zeros(theta.shape))
    for i in range(iter):
        error = (x @ theta.T - y)
        for j in range(parameters):
            term =np.multiply(error, x[:,j] )
            tmp[0,j] =theta[0,j] - (alpha/len(x)) * np.sum(term)

        theta = tmp
        cost[i] = compute_cost(x,y,theta)
    return theta ,cost

theta , cost = gradient_descent(x,y,theta,0.1,100)
print(theta)
print("*****************************")
print(cost)
