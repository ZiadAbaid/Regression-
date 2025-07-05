import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

path = r"C:\MATERIAL\CS 4\data3.txt"
data = pd.read_csv(path)
data.rename(columns={0:'x1',1:'x2' ,2 : 'y'}, inplace=True)

data.insert(0,'x0',value= 1)
cols = data.shape[1]
rows= data.shape[0]

# creating matrices
x=np.matrix(data.iloc[: ,0:cols-1].values)
y=np.matrix(data.iloc[: , cols-1 : cols].values)
theta = np.zeros(cols-1)
print(theta.shape , x.shape)


def sigmoid(x) :
    return 1/(1+np.exp(-x))

def cost(theta,x,y):
    theta = np.matrix(theta)
    cost = -1* (1/len(y)) * np.sum(np.multiply( y ,np.log(sigmoid(x * theta.T)) ) + np.multiply( (1-y) , np.log(1-sigmoid(x * theta.T)) ))
    return cost

def gradient(theta,x,y):
    theta = np.matrix(theta)
    error = (sigmoid(x*theta.T) - y)
    parameters = theta.shape[1]
    grad = np.zeros(parameters)
    for i in range(parameters):
        term =np.multiply(error , x[: ,i] )
        grad[i] = (1/len(y)) * np.sum(term)
    return grad

def accuracy(y_exp,y):
    y = np.array(y).ravel()
    y_exp = np.array(y_exp)
    correct = np.sum(y_exp == y)
    return correct / len(y) * 100

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))

theta_min = np.matrix(result[0])

y_exp = predict(theta_min, x)


print(accuracy(y_exp, y))

part_1 = data[ data.iloc[:,3] ==1 ]
part_2 = data[ data.iloc[:,3] ==0 ]
plt.scatter(part_1['x1'], part_1['x2'],
            marker='o', color='red',  label='Class 1')
plt.scatter(part_2['x1'], part_2['x2'],
            marker='x', color='blue', label='Class 0')


plt.legend()
plt.show()