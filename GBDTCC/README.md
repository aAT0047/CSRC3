# SimpleGradientBoost

This is Simple GradientBoost implementing.  
  
I deliberately added comments for anyone to understand code well.  
  
I refered YOUTUBE,  
[StatQuest with Josh Starmer : Gradient Boost Part 1 (of 4): Regression Main Ideas](https://youtu.be/3CC4N4z3GJc, "youtube link1")  
[StatQuest with Josh Starmer : Gradient Boost Part 2 (of 4): Regression Details](https://youtu.be/2xudPOBz-vs, "youtube link2")  
[StatQuest with Josh Starmer : Gradient Boost Part 3 (of 4): Classification](https://youtu.be/jxuNLH5dXCs, "youtube link3")  
[StatQuest with Josh Starmer : Gradient Boost Part 4 (of 4): Classification Details](https://youtu.be/StWY5QWMXCw, "youtube link4")

### Abstract
I made GradientBoostingRegressor algorithm after studying it.  
To understand it, Refer upper youtube link! My code is just implementing logic introduced in that.  
A GradientBoostingRegressor contains numerous weak regressors based on tree. So I used sklearn.tree.DecisionTreeRegressor.  
( I think that implementing DecisionTreeRegressor is much harder than implementing GradientBoostingRegressor.  
I'll implement DecisionTree later... )


### Usage  
1. The code of my custom GradientBoost algorithm is in **algorithm.py**  
2. To run and To test my GradientBoost algorithm, Run **run_it_testAlgorithm.py** s.t runs and compares it with sklearn.ensemble.GradientBoostingRegressor  
