import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# Part 1
# Please do not modify this seed
np.random.seed(6996)

# error term (reshape it so it's a row vector)
epsilon_vec = np.random.normal(loc = 0, scale = 1.0, size = 500).reshape(500,1)
# X_matrix or regressors or predictiors (500 X 500)
X_mat = np.random.normal(loc = 0, scale = 2.0, size = (500,500))
# Slope
slope_vec = np.random.uniform(low = 1, high = 5, size = 500)
# Simulate Ys
# each col of Y_mat representing one simulation vector: starting with 2 regressors, end with 500
Y_mat = 1 + np.cumsum(X_mat * slope_vec, axis = 1)[:,1:] + epsilon_vec

# Create linear regression object
# It will be used to get the coefficients and pvalue
lhs = Y_mat[:,488]
rhs = sm.add_constant(X_mat[:,:490])
regr =  sm.OLS(lhs, rhs).fit()

# create the function add_reg which will add a number of regressor
def add_reg(num,level=0.05):
    if num < 2:
        print('not enough regressor selected')
        return
    else:
        
        # Regressor Matrix ＋ Constant (RHS)
        X = sm.add_constant(X_mat[:,:num])
        # Dependent Variable Vector (LHS)
        y = Y_mat[:,num-2]
        # Linear regression object
        ols_reg = sm.OLS(y, X).fit()
        # R square (computed in the fit)
        r2 = ols_reg.rsquared
        # confidence interval (choose alpha = level by default = 0.05)
        intv = ols_reg.conf_int(alpha = level, cols = None)[1]
        # (store the rsquared and the confidence interval min max here)
        return r2, intv 
    
        
        

def test_part1():
    print (X_mat)

def test_part1_1():
    print(Y_mat.shape)

def test_part1_2():
    print(Y_mat)

def test_part2():
    print(regr.params)

def test_part2_1():
    print(regr.pvalues)


def test_part3():
    result = list(map(add_reg, range(2,501,100)))
    print(result)


if __name__ == '__main__':
    #func_name = "test_part2"
    func_name = input().strip()
    globals()[func_name]()