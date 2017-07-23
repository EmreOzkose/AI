from knn import NearestNeighbor as knn
from read_data import read_cifar_data
import numpy as np
import time

data_train_x, data_train_y = read_cifar_data()

m = 3500
n = 32*32*3
x = time.time()
train_x = np.array((m,n)); train_x = data_train_x[0:m]
train_y = np.array((m,1)); train_y = data_train_y[0:m]
val_x = np.array((m,n)); val_x = data_train_x[m:2*m]
val_y = np.array((m,1)); val_y = data_train_y[m:2*m]

# use a particular value of k and evaluation on validation data
nn = knn()
nn.train(train_x, train_y)
# here we assume a modified NearestNeighbor class that can take a k as input
Yval_predict = nn.predict(val_x)
Yval_predict = Yval_predict.reshape((m,1))

print(Yval_predict.shape)
print(val_y.shape)

res = np.zeros((m,1))

index = 0
for i in Yval_predict == val_y:
    if i == [False]:
        res[index] = 0
    elif i== [True]:
        res[index] = 1

    index+=1

acc = np.mean(res)
print('accuracy: {}'.format(acc))

y = time.time()
print(y-x)