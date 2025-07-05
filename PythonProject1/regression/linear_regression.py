import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  mpl_toolkits.mplot3d import Axes3D

path = r"C:\MATERIAL\CS 4\data2.txt"

data = pd.read_csv(path,names=['x1','x2','y'] )
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
iter=100*100*50
theta , cost = gradient_descent(x,y,theta,1,iter)
print(theta)
print("*****************************")
f= x @ theta.T
ssrec = np.sum(np.power(y - f ,2))
sstotal = np.sum(np.power(y-y.mean(),2))
r_square = 1 - ssrec/sstotal
print(r_square)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
X,Y = np.meshgrid(x[: , 1],x[: , 2])
Z=x[: ,0] * theta[0,0] + X * theta[0,1] + Y * theta[0,2]


fig= plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X,Y,Z)
plt.show()

plt.subplot(2,1,1)
domain = np.linspace(data.x1.min(),data.x1.max(),100)
f= theta[0,0]+ theta[0,1] * domain
plt.plot(domain,f,'--b')
plt.scatter(data.x1,data.y)
plt.xlabel('x1')
plt.ylabel('y')
plt.subplot(2,1,2)
plt.plot(np.arange(iter),cost,'--r')
plt.xlabel('iteration')
plt.ylabel('cost')


plt.show()