import numpy as np 
import math

def sigmoid(value):
    sig_value=1/(1+math.exp(-value))
    return sig_value

def compute_cost(x,y,reg_lambda=0.1):
    w,b= gradient_descent(x,y)
    cost=0
    m,n=x.shape
    len_w=w.shape[0]
    for i in range(m):
        z=np.dot(x[i],w)+b
        z=sigmoid(z)
        cost += -y[i]*np.log(z) - (1-y[i])*np.log(1-z)

    cost=cost/m
    reg_cost=0
    for j in range(len_w):
        reg_cost+= pow(w[j],2)
    reg_cost=(reg_cost*reg_lambda)/(2*m)
    return cost+reg_cost

def compute_gradient(x, y, w, b,reg_lambda=0): 
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    
    for i in range(m):
        temp_error= sigmoid(np.dot(x[i],w)+b) -y[i]
        for j in range(n):
            dj_dw[j]= dj_dw[j]+ (temp_error*x[i,j])
        dj_db+= temp_error
    
    dj_dw= dj_dw/m + (reg_lambda*w)/m
    dj_db= dj_db/m
    
    return dj_dw, dj_db


def gradient_descent(x,y,learning_rate=0.04,iterations=9000):
    m,n=x.shape
    b=0
    w=np.zeros(n,)   
    for i in range(iterations):
        dj_dw, dj_db= compute_gradient(x,y,w,b)
        w= w-(learning_rate*dj_dw)
        b= b- (learning_rate* dj_db)

    #print("w={}, b={}".format(w,b))
    return w,b

def scaling(x):
    std_dev= np.std(x)
    mean=np.mean(x)
    z_scored= (x-mean)/std_dev
    return z_scored

x=np.array([[0.5,1.5],[1,1],[1.5,0.5],[3,0.5],[2,2],[1,2.5]])
y=np.array([0,0,0,1,1,1])
x=scaling(x)
print(gradient_descent(x,y))
#print(compute_cost(x,y))
