import numpy as np
from sklearn import linear_model

np.random.seed(6996)

# error term
epsilon_vec = np.random.normal(0,1,500).reshape(500,1)
# X_matrix or regressors or predictiors
X_mat = np.random.normal(0,2,size = (500,500))
# Slope
slope_vec = np.random.uniform(1,5,500)
# Simulate Ys
Y_mat = 1 + np.cumsum(X_mat * slope_vec,axis=1)[:,1:] + epsilon_vec
# each col of Y_mat representing one simulation vector: starting with 2 regressors, end with 500



# Question 1: Fit linear regression
def fit_lr(X, Y, N_feature=10, split_ratio=0.7, out_put=True):
    train_size = np.int(len(X) * split_ratio)
    X_train = X_mat[0:train_size, 0:N_feature]
    Y_train = Y_mat[0:train_size, N_feature - 2]
    X_test = X_mat[train_size:, 0:N_feature]
    Y_test = Y_mat[train_size:, N_feature - 2]
    
    from sklearn.metrics import mean_squared_error
    # Create linear regression object
    regr = linear_model.LinearRegression().fit(X_train,Y_train) # Fit training set
    # model evaluation
    mse_in = mean_squared_error(Y_train, regr.predict(X_train)) # Get MSE of training set
    mse_out = mean_squared_error(Y_test, regr.predict(X_test)) # Get MSE of testing set using IS-Params
    if out_put:
        print('Coefficients for first %d predictors: \n' % N_feature, regr.coef_)
    print('\n In-sample Mean Square Error: ', mse_in)
    print('Out-of-sample Mean Square Error: ', mse_out)

    return regr.coef_, mse_out


# Question 2: Ridge
def fit_ridge():
    
    from sklearn.linear_model import RidgeCV
    
    ## subset data
    N_feature = 10
    split_ratio = 0.7
    train_size = np.int(len(X_mat)*split_ratio)
    X_train = X_mat[0:train_size,0:N_feature]
    Y_train = Y_mat[0:train_size,N_feature-2]
    X_test = X_mat[train_size:,0:N_feature]
    Y_test = Y_mat[train_size:,N_feature-2]
    ## Ridge from sklearn
    # You will use ridge with the following parameters:
    alphas = [5**i for i in range(-8,2)]
    cv = 5
    
    # ridgereg = RidgeCV(alpha = alphas).fit(X_train,Y_train)
    # y_pred = ridgereg.predict(X_test)
    ridge_cv = RidgeCV(alphas, cv = 5).fit(X_train,Y_train) # To be completed
    return ridge_cv,X_train,Y_train,X_test,Y_test


# Question 3: Lasso
def fit_lasso(X, Y, N_feature=10, out_put = True):
    X_sub= X_mat[:,0:N_feature]
    Y_sub = Y_mat[:,N_feature-2]
    # Create linear regression object using Lasso
    # To be completed
    from sklearn.linear_model import LassoCV
    from sklearn.metrics import mean_squared_error
    # alphas = [5**i for i in range(-8,2)]
    regr = LassoCV(cv = 4).fit(X_sub,Y_sub)
    mse = mean_squared_error(Y_sub, regr.predict(X_sub))
    if out_put:
        print('Coefficients for first %d predictors: \n' %N_feature,regr.coef_)
    print('Mean Square Error: ', mse)
    print('Best alpha for Lasso',regr.alpha_)
    return regr.coef_, mse

    
def question1():
    ## 10 predictors & 491 predictors
    m10_coef, m10_mse = fit_lr(X_mat, Y_mat)

    ## If we apply 490 predictors, based on the small train sample size, the result would not make sense.
    ## The difference of in/out sample error is a good indicator of overfitting
    v490_coef, v490_mse = fit_lr(X_mat, Y_mat, N_feature=490, out_put=False)

    print(m10_coef[:5], m10_mse )
    print(v490_coef[:5], v490_mse)


def question2():
    ridge_cv,X_train, Y_train, X_test, Y_test = fit_ridge()
    mse_in = np.mean(np.square(Y_train - ridge_cv.predict(X_train)))
    mse_out = np.mean(np.square(Y_test - ridge_cv.predict(X_test)))
    print('best_alpha: ',ridge_cv.alpha_)
    print('Regression Coefficients: \n',ridge_cv.coef_)
    print('In-sample Mean Square Loss: ',mse_in)
    print('Out-of-Sample Mean Square Loss: ', mse_out)


def question3():
    ## 10 predictors & 491 predictors
    m10_coef, m10_mse = fit_lasso(X_mat, Y_mat)

    v490_coef, v490_mse = fit_lasso(X_mat, Y_mat, N_feature=490, out_put=False)

    mark_zeros = np.where(v490_coef == 0.0)

    print(m10_coef[:5], m10_mse)
    print(v490_coef[:5], v490_mse)
    print(mark_zeros[:5])

if __name__ == '__main__':
    func_name = input().strip()
    globals()[func_name]()
