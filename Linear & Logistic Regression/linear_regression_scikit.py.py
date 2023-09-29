import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

scaler=StandardScaler()
x_scaled=scaler.fit_transform(X_train)
#y_scaled=scaler.fit_transform(y_train)

# def scaling(x):
#     std_dev= np.std(x)
#     mean=np.mean(x)
#     z_scored= (x-mean)/std_dev
#     return z_scored

# x_scaled=scaling(X_train)
# y_scaled=scaling(y_train)
sgdr=SGDRegressor(max_iter=10000, learning_rate='constant', eta0=0.01)
sgdr.fit(x_scaled,y_train)
print(sgdr.coef_, sgdr.intercept_,sgdr.n_iter_)