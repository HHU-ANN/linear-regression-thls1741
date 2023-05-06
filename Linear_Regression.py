# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def standard_st(data):
    data_mean=np.mean(data)
    data_std=np.std(data)
    data=(data-data_mean)/data_std
    return data

def data_st(X):
    X=np.apply_along_axis(standard_st,1,X)
    ones=np.ones((X.shape[0],1))
    X=np.hstack((X,ones))
    return X

def data_01(X):
    X=np.apply_along_axis(standard_01,1,X)
    ones=np.ones((X.shape[0],1))
    X=np.hstack((X,ones))
    return X

def standard_01(data):
    min_val=np.min(data)
    max_val=np.max(data)
    scaled_data=(data-min_val)/(max_val-min_val)
    return scaled_data

def ridge(data):
    data=standard_01(data)
    X,y=read_data()
    X=data_01(X)
    w=np.zeros((7,1))
    y_pre=X@w
    alpha=1
    ridgeloss=2*(X.T@X@w-X.T@y+alpha*w)
    w=np.linalg.inv((X.T@X+alpha*np.eye(np.shape((X.T@X))[0])))@X.T@y
    b=w[-1]
    w=w[:-1]
    w=w.reshape(6,1)
    return data@w+b

    
def lasso(data):
    data=standard_st(data)
    X,y=read_data()
    X=data_st(X)
    y=y.reshape(1,404)
    alpha=1000
    beta=0.00045
    w=np.zeros((7,1))
    best=w
    min=365194055
    loss_old=1
    for i in range(100000):
        y_pre=X@w
        mse=np.sum(((X@w)-y.T)@((X@w)-y.T).T)/(np.shape(X)[0])
        l1=alpha*((np.sum(np.abs(w))))
        lassoloss=mse+l1
        dw=X.T@((X@w)-y.T)+alpha*np.sign(w)
        loss_old=lassoloss
        w=w-beta*dw
        if(np.abs(min-loss_old)<0.0001):
            print('终止')
            break
        if(min>=lassoloss):
            min=lassoloss
            best=w
    w=best[0:6,:]
    b=best[6,0]
    print(data@w+b)
    return data@w+b


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y