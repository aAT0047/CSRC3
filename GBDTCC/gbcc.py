import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from math import exp, log
from loss import BernoulliLoss



def sigmoid(x, x_min=-100):
    return 1 / (1 + exp(-x)) if x > x_min else 0
class GradientBoosting:
    
    # params_keys : ['learning_rate', 'n_estimators', 'max_depth', 'max_features']
    # params_keys : ['learning_rate', 'n_estimators', 'max_leaf_nodes', 'max_features', 'random_state', 'verbose']
    
    def __init__(self, Q, T, delta=0.01,subsample= 1.0,params={}):
        # Hyperparameters of GradientBoost
        self.learning_rate = params.get('learning_rate', .6)
        self.n_estimators = params.get('n_estimators', 30)
        self.loss = BernoulliLoss(2)
        # Hyperparameters of Weak Regressor(DecisionTreeRegressor) of GradientBoost
        self.max_leaf_nodes = params.get('max_leaf_nodes', 8)
        self.max_features = params.get('max_features', None)
        self.fn = sigmoid
        # parameters for ...
        self.random_state = params.get('random_state', 42)
        self.verbose = params.get('verbose', 0)
        self.trees = None
        self.lr = 0.2
        self.init_val = None
           # stores all Decisiontree
        self.allLearner = []    #(T,Q)
        self.Alpha_s = []   #(T,Q)
        self.Order_s = []  #(T,Q)
        self.Q = Q
        self.T = T
        self.delta = delta
        self.subsample = subsample
        
    def get_init_val(self, y):
        n = len(y)
        y_sum = sum(y)
        return log((y_sum) / (n - y_sum))


    

   
        
    def gbdfit(self, X, target,leaner):
      
        # STEP 1 : Init model with a constant value, F0(x) = argmin(r)(Σ(loss(y, r)))
        ## r is equal with average of all values of target
        self.init_val = self.get_init_val( target)
        n = len(target)
        y_hat = [self.init_val] * n
        # residuals = self.get_residuals(target, y_hat)
        # print(residuals)
        self.trees = []
        
        
        # STEP 2 : makes and fit Weak predictor (We use DecisionTree) for each step
        for i in range(1, self.n_estimators+1):
           
            mask = 0
            # mask = (np.random.random(size=np.array(X.shape[0])) < self.subsample) if self.subsample < 1.0 else np.ones(np.array(X.shape[0]), dtype=np.bool)
            g = self.loss.negative_gradient(target, y_hat)
          
            leaner.fit(X, g)
            residuals = self.loss.update_terminal_regions(leaner, X, target, y_hat, g, self.lr)
            self.trees.append(leaner)
        print(pd.DataFrame(residuals))
        return residuals,self.trees



    def _predict(self,qq, Xi):
        for tree in self.allLearner[qq]:

            ret = self.init_val + sum(self.lr * tree.predict(np.array(Xi).reshape(1, -1)))
        # print(self.allLearner[qq])
            rr = sigmoid(ret)
      
        return rr

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
      
         
            singleLearner= DecisionTreeRegressor( random_state=self.random_state,max_depth=2,min_samples_split=2)
            rs ,treeLearner= self.gbdfit(X_train, Y[:,qq],singleLearner)
            baseLearner[qq] = treeLearner
            Alpha[qq] = 1
            rs_s[qq] = rs
            X_train = np.hstack((X_train, Y[:,[qq]]))
            # print(pd.DataFrame(X_train) )
        self.allLearner = baseLearner
        # print(pd.DataFrame(self.allLearner))
        self.Alpha_s.append(Alpha)
        return  rs_s

    def predictx(self, X, threshold=0.5):
        
        return [int(self.probe(Xi) >= threshold) for Xi in X]
        

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
    print( Xt)
    Yt = Y_all[test]
    # start_time = time()
 
    params = {'learning_rate':0.5, 'n_estimators':100, 'max_leaf_nodes':16, 'verbose':1}
    mlClassifier3 =GradientBoosting(6,1,params)
    mlClassifier3.fit(X,Y)

  
    
    # prediction3 = mlClassifier3.predict(Xt)
    # print(pd.DataFrame( prediction3))
    y_hat_prob = mlClassifier3.probe(Xt)
    print(pd.DataFrame( y_hat_prob ))
    # model3=evaluate(y_hat_prob, Yt)
    # end_time = time()

   
    # modelsave3.append(model3)

# mean1 = np.mean(modelsave1, axis=0)

# mean3 = np.mean(modelsave3, axis=0)
# print(mean1)

# print(mean3)
# # solveResult(mean)
# t = round(end_time-start_time,1)
# print(t)
from sklearn.metrics import label_ranking_loss
print(label_ranking_loss(Yt,y_score=y_hat_prob ))
from sklearn.metrics import coverage_error
print(coverage_error(Yt,y_score=y_hat_prob ))