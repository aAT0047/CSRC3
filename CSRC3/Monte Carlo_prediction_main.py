# from AdaboostC2 import readData_CV2
# from AdaboostC2 import AdaboostC3
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression as LR
# from mReadData import *
from mEvaluation import evaluate
from operator import itemgetter
import random
import pandas as pd
from mEvaluation import evaluate
from scipy.stats import chi2
from scipy.stats import ncx2
from adacost import AdacostC3
from adacost import readData_CV2
import pprint
pd.set_option('display.max_rows', 20)


class montecar (AdacostC3):

    def __init__(self, Q, T, delta=0.01):
        super().__init__(Q, T, delta)
 

    def Hosmer_Lemeshow_testcg(self,data, Q=79):
                                                                            
        data = data.sort_values('y_hat')
        # print(data)
        data['Q_group'] = pd.qcut(data['y_hat'], Q,duplicates='drop')
        # print(data['Q_group'])
        y_p = data['y'].groupby(data.Q_group).sum()
        y_total = data['y'].groupby(data.Q_group).count()
        y_n = y_total - y_p
        # print(y_p)
        y_hat_p = data['y_hat'].groupby(data.Q_group).sum()
        y_hat_total = data['y_hat'].groupby(data.Q_group).count()
        y_hat_n = y_hat_total - y_hat_p
        # print(pd.concat([y_p,y_n,y_hat_p,y_hat_n],axis=1))
        hltest = (((y_p - y_hat_p)**2 / y_hat_p) + ((y_n - y_hat_n)**2 / y_hat_n)).sum()
        l = hltest-(Q-2)
        if l<0:
            pval = 1-chi2.cdf(hltest,Q-2)
        else:
            pval = 1-ncx2.cdf(hltest,Q-2,hltest-(Q-2))       
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
    Yt = pd.DataFrame(Yt)      
    classifier = montecar(5,10)
    classifier.induce(X,Y)
    prediction = pd.DataFrame(classifier.test(Xt))
    h = 0
    p= np.zeros([1,5])
    for i in range (0,5):
        data =pd.concat([ prediction.iloc[:,i],Yt.iloc[:,i]],axis=1)
        data.columns= ['y_hat','y']
        h1,pp=classifier.Hosmer_Lemeshow_testcg(data)
        h+=h1
        p[:,i] = pp
    h = h/5
    # print(h)
    # print(p)
    modelsave3 = modelsave3+h
    modelpp  += p
    # print(p)
import numpy as np
from sklearn.metrics import label_ranking_loss
# print(label_ranking_loss(Yt, prediction))
# print("mean--",modelsave3/5)
# print("mean--",modelpp/5)


k_fold, X_all, Y_all = readData_CV2()
modelsave3 = 0
modelpp = np.zeros([1,5])
for train, test in k_fold.split(X_all, Y_all):
    X = X_all[train]
    Y = Y_all[train]
    Xt = X_all[test]
    Yt = Y_all[test]
    Yt = pd.DataFrame(Yt)      
    classifier = montecar(5,10)
    classifier.induce(X,Y)
    prediction = pd.DataFrame(classifier.test(Xt))
    from joblib import dump

    # 假设你的模型是通过classifier.induce(X,Y)训练的
    model = classifier.induce(X, Y)

    # 保存模型到文件系统
    dump(model, 'model.joblib')

    
    h = np.zeros([1,5])
    p= np.zeros([1,5])
    for i in range (0,5):
        data =pd.concat([ prediction.iloc[:,i],Yt.iloc[:,i]],axis=1)
        data.columns= ['y_hat','y']
        h1,pp=classifier.Hosmer_Lemeshow_testcg(data)
        h[:,i]=h1
        p[:,i] = pp
    # h = h/5
    # print(h)
    # print(p)
    modelsave3 += h
    modelpp  += p
    # print(p)
import numpy as np
from sklearn.metrics import label_ranking_loss
# print(label_ranking_loss(Yt, prediction))
# print("mean--",modelsave3/5)
# print("mean--",modelpp/5)
# pd.options.display.max_columns = 5

prediction.to_csv(r'C:\Users\1\Desktop\data\p.csv', index=False)
Yt.to_csv(r'C:\Users\1\Desktop\data\yy.csv', index=False)