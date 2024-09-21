from algorithm import GradientBoostingRegressor_MarsMan13
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings(action='ignore')

# Load data for regressor problem
boston = load_boston()
boston_data = boston.data
boston_target = boston.target
X_train, X_test, y_train, y_test = train_test_split(boston_data, boston_target, test_size=0.1)

# TEST OUR GRADIENT BOOST ALGORITHM, GradientBoostingRegressor_MarsMan13
params = {'learning_rate':0.1, 'n_estimators':300, 'max_leaf_nodes':16, 'verbose':1}
gbm = GradientBoostingRegressor_MarsMan13(params)
gbm.fit(X_train, y_train)
pred = gbm.predict(X_test)
print(pred )
mse = mean_squared_error(y_test, pred)

# TEST sklearn.ensemble.GradientBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
gbm_ = GradientBoostingRegressor(learning_rate=0.1, n_estimators=300, max_leaf_nodes=16)
gbm_.fit(X_train, y_train)
pred_ = gbm_.predict(X_test)
mse_ = mean_squared_error(y_test, pred_)

print("==== SHOW RESULT ====")
print("Our algo' mse : {0}, Sklearn algo's mse : {1}".format(round(mse,3), round(mse_,3)))