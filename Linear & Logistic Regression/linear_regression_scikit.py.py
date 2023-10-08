import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

X_train = np.array([1,2,3,4,5])
y_train = np.array([3.66,3.62,3.77,3.88,3.81])

scaler=StandardScaler()
#x_scaled=scaler.fit_transform(X_train)
#y_scaled=scaler.fit_transform(y_train)

# def scaling(x):
#     std_dev= np.std(x)
#     mean=np.mean(x)
#     z_scored= (x-mean)/std_dev
#     return z_scored

# x_scaled=scaling(X_train)
# y_scaled=scaling(y_train)
sgdr=SGDRegressor(max_iter=10000, learning_rate='constant', eta0=0.01)
sgdr.fit(X_train,y_train)
print(sgdr.predict(8))