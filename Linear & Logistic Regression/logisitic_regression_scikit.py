import numpy as np 
from sklearn.linear_model import LogisticRegression
x=np.array([[0.5,1.5],[1,1],[1.5,0.5],[3,0.5],[2,2],[1,2.5]])
y=np.array([0,0,0,1,1,1])

lr_model=LogisticRegression()
lr_model.fit(x,y)
print(lr_model.coef_,lr_model.intercept_)
print(lr_model.n_iter_)