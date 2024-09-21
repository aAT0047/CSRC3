from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
# from mReadData import *
# from mEvaluation import evaluate
from Base import *
import random
from mEvaluation import evaluate
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression as LR
from scipy.stats import chi2
from scipy.stats import ncx2
from scipy.stats import chi2


def randorder(Q):
    return np.array(random.sample(range(Q),Q))
    
numBase =10
modelsave1 = []
def readData_CV2(CV=5):
        data_X= pd.read_csv(r"C:\Users\1\Desktop\data\emer_dum.csv").iloc[:,1::]
        X=data_X.values
        data_Y=pd.read_csv(r"C:\Users\1\Desktop\data\simulativeLabel_6.csv").iloc[:,1::]
        Y = data_Y.values
        # print(X)
        # print(Y)

        # s = data_Y[:,0]
        # list1= []
        # list0 = []
        # dataset0=[]
        # y0= []
        # dataset1= []
        # y1=[]
        # for i in range(len(s)):
        #     if(s[i] == 1):
        #         list1.append(i)
        #     else:
        #         list0.append(i)
        # for j in list1[0:500]:
        #     y0.append(data_Y[j,:])
        #     dataset0.append(X[j,:])
        # for l in list0[0:500]:
        #     y1.append(data_Y[l,:])
        #     dataset1.append(X[l,:])

        # Y = pd.concat([pd.DataFrame(y0),pd.DataFrame(y1)],axis=0).reset_index().iloc[:,1::].values   
        # X= pd.concat([pd.DataFrame(dataset0),pd.DataFrame(dataset1)],axis=0).reset_index().iloc[:,1::].values     
        k_fold = KFold(n_splits=CV,shuffle=True,random_state=0)
        return k_fold, X, Y 

def Hosmer_Lemeshow_testcg(data, Q=79):
  
        data = data.sort_values('y_hat')
        print(data)
        data['Q_group'] = pd.qcut(data['y_hat'], Q,duplicates='drop')
        # print(data['Q_group'])
        y_p = data['y'].groupby(data.Q_group).sum()
        y_total = data['y'].groupby(data.Q_group).count()
        y_n = y_total - y_p
        # print(y_p)
        y_hat_p = data['y_hat'].groupby(data.Q_group).sum()
        y_hat_total = data['y_hat'].groupby(data.Q_group).count()
        y_hat_n = y_hat_total - y_hat_p
        print(pd.concat([y_p,y_n,y_hat_p,y_hat_n],axis=1))
        # tt = pd.concat([y_p,y_n,y_hat_p,y_hat_n],axis=1)
        # tt.to_csv(r"C:\Users\1\Desktop\tt.csv", index=False, encoding='utf-8')
        hltest = (((y_p - y_hat_p)**2 / y_hat_p) + ((y_n - y_hat_n)**2 / y_hat_n)).sum()
        print((((y_p - y_hat_p)**2 / y_hat_p) + ((y_n - y_hat_n)**2 / y_hat_n)))
        l = hltest-(Q-2)
        if l<0:
            pval = 1-chi2.cdf(hltest,Q-2)
        else:
            pval = 1-ncx2.cdf(hltest,Q-2,hltest-(Q-2))       
        # print(hltest)
        # print(hltest)
        return hltest ,pval
k_fold, X_all, Y_all = readData_CV2()
modelsave3 = 0
modelpp = np.zeros([1,5])
for train, test in k_fold.split(X_all, Y_all):
    X = X_all[train]
    Y = Y_all[train]
    Xt = X_all[test]
    Yt = Y_all[test]
        
    Yt =pd.DataFrame(Yt)
        # Y = fill1(Y)
    start_time = time()
    ensembleLearner = []
    for t in range(numBase):
        mlClassifier = CC(order=[0,4,1,2,3])
        mlClassifier.fit(X,Y)
            # mlClassifier.predict_proba
        ensembleLearner.append(mlClassifier)
      
        probe = np.zeros(np.shape(Yt))
    for t in range(numBase):
        probe += ensembleLearner[t].predict_proba(Xt)
        probe/= numBase
    probe = pd.DataFrame(probe)
    h =0
    p= np.zeros([1,5])
    for i in range (0,5):
        data =pd.concat([ probe.iloc[:,i],Yt.iloc[:,i]],axis=1)
        data.columns= ['y_hat','y']
        # data.to_csv(r"C:\Users\1\Desktop\dd.csv", index=False, encoding='utf-8')
        h1,pp =Hosmer_Lemeshow_testcg(data)
        h+=h1
        p[:,i] = pp
    h = h/5
    modelsave3 = modelsave3+h
    modelpp  += p
print("mean--",modelsave3/5)
print("mean--",modelpp/5)

k_fold, X_all, Y_all = readData_CV2()
modelsave3 = 0
modelpp = np.zeros([1,5])
for train, test in k_fold.split(X_all, Y_all):
    X = X_all[train]
    Y = Y_all[train]
    Xt = X_all[test]
    Yt = Y_all[test]
        
    Yt =pd.DataFrame(Yt)
        # Y = fill1(Y)
    start_time = time()
    ensembleLearner = []
    for t in range(numBase):
        mlClassifier = CC(order=[0,4,1,2,3])
        mlClassifier.fit(X,Y)
            # mlClassifier.predict_proba
        ensembleLearner.append(mlClassifier)
      
        probe = np.zeros(np.shape(Yt))
    for t in range(numBase):
        probe += ensembleLearner[t].predict_proba(Xt)
        probe/= numBase
    probe = pd.DataFrame(probe)
    h =np.zeros([1,5])
    p= np.zeros([1,5])
    for i in range (0,5):
        data =pd.concat([ probe.iloc[:,i],Yt.iloc[:,i]],axis=1)
        data.columns= ['y_hat','y']
        h1,pp =Hosmer_Lemeshow_testcg(data)
        h[:,i]=h1
        p[:,i] = pp
    # h = h/5
    modelsave3 = modelsave3+h
    modelpp  += p
print("mean--",modelsave3/5)
print("mean--",modelpp/5)
probe.to_csv(r"C:\Users\1\Desktop\data\pecc.csv",index= False)
Yt.to_csv(r"C:\Users\1\Desktop\data\yecc.csv",index= False)