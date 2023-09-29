import numpy as np

def compute_cost(x, y): 

    m = x.shape[0] 
    total_cost = 0
    w,b=gradient_descent(x,y)
    
    for i in range(m):
        temp= (w*x[i])+b -y[i]
        total_cost += (temp**2)
    
    total_cost= total_cost/(2*m)
    return total_cost

def compute_gradient(x, y, w, b): 
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):
        temp_error=w*x[i]+b-y[i]
        dj_dw_i= temp_error*x[i]
        dj_db_i= temp_error
        dj_dw+=dj_dw_i
        dj_db+=dj_db_i
    
    dj_dw= dj_dw/m
    dj_db= dj_db/m
    return dj_dw, dj_db


def gradient_descent(x,y,learning_rate=0.1):
    w,b=0,0    
    while True:
        dj_dw, dj_db= compute_gradient(x,y,w,b)
        w_initial, b_initial=w,b
        w= w-(learning_rate*dj_dw)
        b= b- (learning_rate* dj_db)
        if abs(w_initial-w)<=1e-9 and abs(b_initial-b)<1e-9:
            break
    print("w={}, b={}".format(w,b))
    return w,b


x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])
#print(compute_cost(x,y))
print(gradient_descent(x,y))