# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def sst(data):
    m=np.mean(data)
    std=np.std(data)
    data=(data-m)/std
    return data

def dst(X):
    X=np.apply_along_axis(sst,1,X)
    gt=np.ones((X.shape[0],1))
    X=np.hstack((X,gt))
    return X

def d01(X):
    X=np.apply_along_axis(s01,1,X)
    gt=np.ones((X.shape[0],1))
    X=np.hstack((X,gt))
    return X

def s01(data):
    mi=np.min(data)
    ma=np.max(data)
    sca=(data-mi)/(ma-mi)
    return sca

def ridge(data):
    data=s01(data)
    X,y=read_data()
    X=d01(X)
    w=np.zeros((7,1))
    pre=X@w
    alpha=1
    rloss=2*(X.T@X@w-X.T@y+alpha*w)
    w=np.linalg.inv((X.T@X+alpha*np.eye(np.shape((X.T@X))[0])))@X.T@y
    b=w[-1]
    w=w[:-1]
    w=w.reshape(6,1)
    return data@w+b

    
def lasso(data):
    data=sst(data)
    X,y=read_data()
    X=dst(X)
    y=y.reshape(1,404)
    alpha=1000
    beta=0.00045
    w=np.zeros((7,1))
    best=w
    min=365194055
    old=1
    for i in range(100000):
        pre=X@w
        mse=np.sum(((X@w)-y.T)@((X@w)-y.T).T)/(np.shape(X)[0])
        l1=alpha*((np.sum(np.abs(w))))
        lloss=mse+l1
        dw=X.T@((X@w)-y.T)+alpha*np.sign(w)
        old=lloss
        w=w-beta*dw
        if(np.abs(min-old)<0.0001):
            print('终止')
            break
        if(min>=lloss):
            min=lloss
            best=w
    w=best[0:6,:]
    b=best[6,0]
    print(data@w+b)
    return data@w+b


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y