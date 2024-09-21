import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from  sklearn.linear_model import LogisticRegression as LR
from math import exp, log
from mEvaluation import evaluate
def sigmoid(x, x_min=-100):
    return 1 / (1 + exp(-x)) if x > x_min else 0
class GradientBoosting:
    
    # params_keys : ['learning_rate', 'n_estimators', 'max_depth', 'max_features']
    # params_keys : ['learning_rate', 'n_estimators', 'max_leaf_nodes', 'max_features', 'random_state', 'verbose']
    
    def __init__(self, Q, T, delta=0.01):
        # Hyperparameters of GradientBoost
        # self.learning_rate = params.get('learning_rate', .1)
        self.n_estimators =50
        self.max_depth = None
        self.min_samples_split = None
        # Hyperparameters of Weak Regressor(DecisionTreeRegressor) of GradientBoost
        self.max_leaf_nodes =2
        self.max_features =None
        self.fn = sigmoid
        # parameters for ...
        self.random_state = 42
        self.verbose = 0
        self.trees = None
        self.lr = 0.6
        self.init_val = None
           # stores all Decisiontree
        self.allLearner = []    #(T,Q)
        self.Alpha_s = []   #(T,Q)
        self.Order_s = []  #(T,Q)
        self.Q = Q
        self.T = T
        self.delta = delta
        self.y_hat = 0
    
    def get_init_val(self, y):
        n = len(y)
        y_sum = sum(y)
        return log((y_sum) / (n - y_sum))

    def get_residuals(self, y, y_hat):
        return [yi- self.fn(y_hat_i )for yi, y_hat_i in zip(y, y_hat)]
    @staticmethod
    def del_loss(observed, predicted):
        return observed - predicted # return Residual
    
    def get_score(self, idxs, y_hat, residuals):
        numerator = denominator = 0
        for idx in idxs:
            numerator += residuals[idx]
            denominator += y_hat[idx] * (1 - y_hat[idx])

        return numerator / denominator

   
    def gbdfit(self, data, target,leaner,lr):
      
        # STEP 1 : Init model with a constant value, F0(x) = argmin(r)(Σ(loss(y, r)))
        ## r is equal with average of all values of target
        self.init_val = self.get_init_val( target)
        print(self.init_val)
        n = len(target)
        y_hat = [self.init_val] * n
        residuals = np.array(self.get_residuals(target, y_hat)) 
        # print(residuals)
        self.trees = []
        self.lr = lr
        leaner.fit(data, residuals.astype('int'))
        self.trees.append(leaner)
        # STEP 2 : makes and fit Weak predictor (We use DecisionTree) for each step
        for i in range(0,  self.n_estimators):
            
            ret = leaner.predict(data)
            y_hat=self.lr*ret+y_hat
            # y_hat= [sigmoid(zi)for zi in z]
            # Update scores of tree leaf nodes
            leaner.fit(data, y_hat)
            # Update y_hat
            # y_hat = [y_hat_i + lr * res_hat_i for y_hat_i, res_hat_i in zip(y_hat, tree.predict(X))]
            # Update residuals
            residuals = self.get_residuals(target, y_hat)
           
            self.trees.append(leaner)
            for j in range (len(residuals)):
                if residuals[j]<=0:
                    residuals[j]=residuals[j]

            if(np.mean(abs(residuals))  <=0.0001) :
                break

        # print(pd.DataFrame(self.trees))
        print(pd.DataFrame(residuals))
        self.y_hat = y_hat
        return residuals,self.trees

    def _predict(self,qq, Xi):
        # print(pd.DataFrame(self.allLearner[qq]))
        # for tt in self.allLearner[qq]:
         
        #     ret =self.lr * tt.predict(np.array(Xi).reshape(1, -1))
        #     rr = self.init_val+ret
        # print(self.allLearner[qq])
        ret = self.init_val + sum(self.lr * tree.predict(np.array(Xi).reshape(1, -1)) for tree in self.allLearner[qq])
    #     
        
        # print(ret)   
        return self.fn(ret)

    def probe(self,Xt):
        Xt = np.array(Xt)

        prob = np.ones((len(Xt ),self.Q))
        for label in range(len(self.Order_s)): 
            
            tt = self.Order_s[label]
            probe =[self._predict(label,Xi) for Xi in Xt]
            prob[:,tt] = probe
            # print(np.array(probe).reshape(1, -1).shape)
            # print(Xt.shape)
            # prediction =  self.allLearner[label].predictx(Xt)
            Xt = np.hstack(([Xt,  np.transpose(np.array(probe).reshape(1, -1))]))
          
        return prob
    def fit(self, X, Y):
        order = [0,1,2,3,4,5]
        # order = [0,1]
        ok=[] # the indexes of exactly classificated labels
        for t in range(self.T):
            rss= self.trainCC(X, Y, order, ok)
            # print(Dst_s)
            # ok = np.argwhere(np.array(rss[:,1])<self.delta).flatten()
            print("ok")
            print(ok)
            # print(pd.DataFrame(rss))
            # indices, L_sorted = zip(*sorted(enumerate(np.array(rss[:,1])), key=itemgetter(1)))
            # order = np.array(indices)
            # print (type(order))
            # order = self.neworder(martix_1,error_s)
            print(t,"order")
            print(order)

    def trainCC(self, X, Y,  order, ok):
        self.Order_s = order
        order = order[len(ok):]
        X_train = np.array(X)
        if(len(ok)>0):
            for q in ok:
                X_train = np.hstack((X_train, Y[:,[q]]))
        Alpha = ['']*self.Q
        baseLearner = ['']*self.Q
        rs_s = ['']*self.Q
        
        for qq in order:
      
         
            singleLearner= LR(penalty='l1',solver='liblinear')
            rs ,treeLearner= self.gbdfit(X_train, Y[:,qq],singleLearner,self.lr)
            baseLearner[qq] = treeLearner
            Alpha[qq] = 1
            rs_s[qq] = rs
            X_train = np.hstack((X_train, Y[:,[qq]]))
            # print(pd.DataFrame(X_train) )
        self.allLearner = baseLearner
        # print(pd.DataFrame(self.allLearner))
        self.Alpha_s.append(Alpha)
        return  rs_s

    def predictx(self, Xt, threshold=0.5):
        Xt = np.array(Xt)
        predt = np.ones((len(Xt ),self.Q))
        for label in range(len(self.Order_s)): 
            tt = self.Order_s[label]
            pred = [int(self._predict(label,Xi) >= threshold) for Xi in Xt]
            predt[:,tt] = pred
            Xt = np.hstack(([Xt,  np.transpose(np.array(pred).reshape(1, -1))]))
        return predt
        

def readData_CV2(CV=5):
        data_X= pd.read_csv(r"C:\Users\1\Desktop\data\emer_dum.csv").iloc[:,2::]
        X=data_X.values
        data_Y=pd.read_csv(r"C:/Users/1/Desktop/data/simulativeLabel_6csv").iloc[:,1::].values
        s = data_Y[:,0]
        list1= []
        list0 = []
        dataset0=[]
        y0= []
        dataset1= []
        y1=[]
        for i in range(len(s)):
            if(s[i] == 1):
                list1.append(i)
            else:
                list0.append(i)
        for j in list1[0:900]:
            y0.append(data_Y[j,:])
            dataset0.append(X[j,:])
        for l in list0[0:900]:
            y1.append(data_Y[l,:])
            dataset1.append(X[l,:])

        Y = pd.concat([pd.DataFrame(y0),pd.DataFrame(y1)],axis=0).reset_index().iloc[:,1::].values   
        X= pd.concat([pd.DataFrame(dataset0),pd.DataFrame(dataset1)],axis=0).reset_index().iloc[:,1::].values     
        k_fold = KFold(n_splits=CV,shuffle=True,random_state=0)
        return k_fold, X, Y
k_fold, X_all, Y_all = readData_CV2()

modelsave1 = []
modelsave3 = []
for train, test in k_fold.split(X_all, Y_all):
    # print(pd.DataFrame(X_all))
    # print(pd.DataFrame(X_all[:, ~np.isnan(X_all).all(axis=0)]))
    X = X_all[train]
    # print(pd.DataFrame( X ))
    Y = Y_all[train]
   
    Xt = X_all[test]
   
    Yt = Y_all[test]
    # start_time = time()
 
    params = {'learning_rate':0.6, 'n_estimators':50, 'max_leaf_nodes':16, 'verbose':1}
    mlClassifier3 =GradientBoosting(6,1)
    mlClassifier3.fit(X,Y)

  
    
    prediction3 = mlClassifier3.predictx(Xt)
    print(pd.DataFrame( prediction3))
    y_hat_prob = mlClassifier3.probe(Xt)
    print(pd.DataFrame( y_hat_prob))
    # print(pd.DataFrame( y_hat_prob ))
    model3=evaluate(y_hat_prob, Yt)
    # end_time = time()

   
    modelsave3.append(model3)

# mean1 = np.mean(modelsave1, axis=0)

mean3 = np.mean(modelsave3, axis=0)
# print(mean1)

print(mean3)
# # solveResult(mean)
# t = round(end_time-start_time,1)
# print(t)
from sklearn.metrics import label_ranking_loss
print(label_ranking_loss(Yt,y_score=y_hat_prob ))
from sklearn.metrics import coverage_error
print(coverage_error(Yt,y_score=y_hat_prob ))
from sklearn.metrics import accuracy_score
print(accuracy_score(Yt,prediction3))
from sklearn.metrics import hamming_loss
print(hamming_loss(Yt,prediction3))
pd.DataFrame( prediction3).to_csv(r"C:\Users\1\Desktop\data\pred.csv")
pd.DataFrame( Yt).to_csv(r"C:\Users\1\Desktop\data\yt.csv")