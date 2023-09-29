import numpy as np

def compute_cost(x, y,reg_lambda=0.0): 

    m,n = x.shape 
    total_cost = 0
    w,b=gradient_descent(x,y)
    len_w=w.shape[0]

    for i in range(m):
        temp= np.dot(w,x[i])+b -y[i]
        total_cost += (temp**2)
    
    total_cost= total_cost/(2*m)

    regular_cost=0
    for j in range(len_w):
        regular_cost+=(w[j]*w[j])
    
    regular_cost= (reg_lambda*regular_cost)/(2*m)

    return total_cost+regular_cost

def compute_gradient(x, y, w, b): 
    m,n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    
    for i in range(m):
        temp_error= np.dot(x[i],w)+b -y[i]
        for j in range(n):
            dj_dw[j]= dj_dw[j]+ np.dot(temp_error,x[i,j]) 
        dj_db+= temp_error
    
    dj_dw= dj_dw/m
    dj_db= dj_db/m

    return dj_dw, dj_db


def gradient_descent(x,y,learning_rate=0.01,iterations=1000,reg_lambda=0.1):
    m,n=x.shape
    b=0
    w=np.zeros(n,)   
    for i in range(iterations):
        dj_dw, dj_db= compute_gradient(x,y,w,b)
        dj_dw+= dj_dw + (reg_lambda*w)/m
        w= w-(learning_rate*dj_dw)
        b= b- (learning_rate* dj_db)

    print("w={}, b={}".format(w,b))
    return w,b


def scaling(x):
    std_dev= np.std(x)
    mean=np.mean(x)
    z_scored= (x-mean)/std_dev
    return z_scored


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
X_train=scaling(X_train)
#y_train=scaling(y_train)
w,b=gradient_descent(X_train,y_train)
#print(np.dot(w,X_train[2])+b)
print(compute_cost(X_train,y_train))
