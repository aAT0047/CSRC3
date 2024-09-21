from time import time
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression as LR
# from mReadData import *
from mEvaluation import evaluate
from operator import itemgetter
import random
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
class AdacostC3():
    def __init__(self, Q, T, delta=0.01):
        self.Q = Q
        self.T = T
        self.delta = delta
        self.allLearner = []    #(T,Q)
        self.Alpha_s = []   #(T,Q)
        self.Order_s = []   #(T,Q)
    def induce(self, X, Y):
        Dst_s = np.ones((self.Q, len(X)))   # initial distribution: Uniform distribution
        order = randorder(self.Q)   # initial order of classifiers chain: random
        order = [0,4,1,2,3]
        order = randorder(5)
        ok = [] # the indexes of exactly classificated labels
        for t in range(self.T):
            Dst_s,error_s = self.trainCC(X, Y, Dst_s, order, ok)
            # print(pd.DataFrame(Dst_s))
            ok = np.argwhere(np.array(error_s)<self.delta).flatten()
            # print('222',ok)
            indices, L_sorted = zip(*sorted(enumerate(np.array(error_s)), key=itemgetter(1)))
            # order = np.array(indices)
            # print(order)
    def trainCC(self, X, Y, Dst_s, order, ok):
        self.Order_s.append(order)
        order = order[len(ok):]
        X_train = np.array(X)
        if(len(ok)>0):
            for q in ok:
                X_train = np.hstack((X_train, Y[:,[q]]))
        Alpha = ['']*self.Q
        baseLearner = ['']*self.Q
        Dst_s2 = ['']*self.Q
        error_s = np.zeros(self.Q)
        for qq in order:
            # singleLearner = svm.SVC(probability=True,C=1.0, kernel='rbf',gamma='scale')
            # singleLearner = RandomForestClassifier(n_estimators=10, random_state=42,max_depth=5,min_samples_split=2)
            singleLearner =LR(penalty='l2',C=0.01,solver= 'liblinear')
            singleLearner.fit(X_train, Y[:,qq], Dst_s[qq])
            baseLearner[qq] = singleLearner
            alpha,Dst2,error = self.boosting_a(X_train, Y[:,qq], Dst_s[qq], singleLearner)
            Alpha[qq] = alpha
            Dst_s2[qq] = Dst2
            error_s[qq] = error
            # print(error_s)
            X_train = np.hstack((X_train, Y[:,[qq]]))
        self.allLearner.append(baseLearner)
        self.Alpha_s.append(Alpha)
        # print(pd.DataFrame)
        return Dst_s2, error_s
    def boosting_a(self, X, Y_a, Dst, learner):
        tmp1 = np.int32(np.round(learner.predict(X)))
        # print(np.shape( Y_a))
        result = np.array(tmp1!=Y_a)
        # print(result.shape)
        error = sum(result*Dst)/len(X)
        if(error>0.5):
            return 0,Dst,error
        if(error < self.delta):
            return 0,np.ones(len(X)),error #np.ones(len(X))
        alpha = 0.5*np.log((1+error)/(1-error))
        # alpha = 0.5*np.log((1-error)/(error))
        # print(pd.DataFrame( self._beta(Y_a, tmp1)))
        # Dst2 =Dst*np.exp(-np.multiply((result-0.5)*2*alpha,self._beta(Y_a, tmp1)))
        Dst2 =np.multiply(Dst*np.exp(-((result-0.5)*2*alpha)),self._beta(Y_a, tmp1))
        # Dst2 =Dst*np.exp(-((result-0.5)*2*alpha))
        # print(pd.DataFrame(Dst2))
        # z = min(Dst*np.exp(-np.multiply((result-0.5)*2*alpha,self._beta(Y_a, tmp1))))*sum(result)+max(Dst*np.exp(-np.multiply((result-0.5)*2*alpha,self._beta(Y_a, tmp1)))) *(np.shape(Y_a)[0]-sum(result))
        Dst3 = self.distribution_adj(Dst2)
        # Dst3 = Dst2/z
        # print(pd.DataFrame(Dst3))

        return alpha,Dst3,error
    def distribution_adj(self, Dst):
        gap = min(Dst)
        if(gap<=0):
            print('dst error!!!')
            Dst = Dst - gap + 0.01
        ssum = sum(Dst)
        Dst = Dst * len(Dst)
        Dst = Dst/ssum
        return Dst
    def test(self, Xt):
        Alpha_weights = self.get_alphaweights()
        # print('1111111111111111111111111',pd.DataFrame(Alpha_weights))
        prediction = np.zeros((self.Q,len(Xt)))
        prediction_aLabel = ['']*self.Q
        # saveMat(Alpha_weights)
        for tt in range(self.T):
            # print('base round:', tt)
            Xt_train = np.array(Xt)
            prediction_t = np.zeros((self.Q,len(Xt)))
            for qq in self.Order_s[tt]:
                if(Alpha_weights[tt][qq]==0):
                    Xt_train = np.hstack((Xt_train, np.reshape(prediction_aLabel[qq], (-1, 1))))
                    continue
                # print(Alpha_weights[tt][qq], np.shape(Xt_train))
                # print(tt,qq)
                prediction_a = self.allLearner[tt][qq].predict_proba(Xt_train)[:,1]
                # print(pd.DataFrame( prediction_a))
                prediction_aLabel[qq] = prediction_a
                prediction_t[qq] = np.array(prediction_a) * Alpha_weights[tt][qq]
                if(Alpha_weights[tt][qq]<0):
                    prediction_t[qq] = -prediction_t[qq]
                Xt_train = np.hstack((Xt_train, np.reshape(prediction_a, (-1, 1))))
            prediction = prediction +np.array(prediction_t)
        # print(pd.DataFrame( prediction ))
        return np.transpose(prediction)

 

        
      
    
    def get_alphaweights(self): #adjust weight
        Alpha_weights = np.zeros((self.T, self.Q))
        for i in range(self.T):
            for j in range(self.Q):
                if(self.Alpha_s[i][j]!=''):
                    Alpha_weights[i][j] = self.Alpha_s[i][j]
        for j in range(self.Q):
            if(Alpha_weights[0][j] == 0):
                Alpha_weights[0][j] = 1
        Alpha_weights = np.transpose(Alpha_weights)
        for i in range(self.Q):
            Alpha_weights[i] = Alpha_weights[i]/sum(np.abs(Alpha_weights[i]))
        Alpha_weights = np.array(np.transpose(Alpha_weights))
        return Alpha_weights
   
    def predict(self, Xt):
        Alpha_weights = self.get_alphaweights()
        prediction = np.zeros((self.Q,len(Xt)))
        prediction_aLabel = ['']*self.Q
        # saveMat(Alpha_weights)
        for tt in range(self.T):
            # print('base round:', tt)
            Xt_train = np.array(Xt)
            prediction_t = np.zeros((self.Q,len(Xt)))
            for qq in self.Order_s[tt]:
                if(Alpha_weights[tt][qq]==0):
                    Xt_train = np.hstack((Xt_train, np.reshape(prediction_aLabel[qq], (-1, 1))))
                    continue
                # print(Alpha_weights[tt][qq], np.shape(Xt_train))
                # print(tt,qq)
                prediction_a = self.allLearner[tt][qq].predict(Xt_train)
                # print(pd.DataFrame( prediction_a))
                prediction_aLabel[qq] = prediction_a
                prediction_t[qq] = np.array(prediction_a) * Alpha_weights[tt][qq]
                if(Alpha_weights[tt][qq]<0):
                    prediction_t[qq] = -prediction_t[qq]
                Xt_train = np.hstack((Xt_train, np.reshape(prediction_a, (-1, 1))))
            prediction = prediction + np.array(prediction_t)
           
        return np.transpose(prediction)
      
      
       #  新定义的代价调整函数
    def _beta(self, y, y_hat):
        res = []
        for i in zip(y, y_hat):
            if i[0] == i[1]:
                res.append(0.5)   # 正确分类，系数保持不变，按原来的比例减少
            elif i[0] == 1 and i[1] == 0:
                res.append(1.5)  # 实际得病，但是未预测出来，按比例增加
            elif i[0] == 0 and i[1] == 1:
                res.append(1)  # 实际未得病，但是预测得病，比例不变
            else:
                print(i[0], i[1])

        return np.array(res)






def randorder(Q):
    return np.array(random.sample(range(Q),Q))
    # return np.arange(Q)
def logmat(mat):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            print(type(mat[i][j]), end=' ')
        print()
def fill1(Y):
    Y = np.array(Y)
    for j in range(np.shape(Y)[1]):
        if(np.sum(Y[:,j])==0):
            Y[0][j] = 1
    return Y
def readData_CV2(CV=5):
        data_X= pd.read_csv(r"C:\Users\1\Desktop\data\emer_dum.csv").iloc[:,1::]
        X=data_X.values
        data_Y=pd.read_csv(r"C:/Users/1/Desktop/data/simulativeLabel_6.csv").iloc[:,1::]
        Y = data_Y.values
        # print(data_X)
        # print(data_Y)

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
        return k_fold,  X, Y 
# k_fold, X_all, Y_all = readData_CV2()

