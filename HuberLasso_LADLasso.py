import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
import pandas as pd
import optuna
from optuna.trial import Trial
optuna.logging.set_verbosity(optuna.logging.FATAL)
import warnings
warnings.filterwarnings("ignore")
from functools import partial

utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
import numpy as np
packnames = np.array(['MTE'])

names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(names_to_install)
#utils.install_packages('MTE')
mte = rpackages.importr('MTE')

def HuberLasso(df, Lambda, c):
    nrow = df.shape[0]
    ncol = df.shape[1]
    returns = df.values
    returns = [i for j in returns for i in j]
    r_returns = robjects.FloatVector(returns)
    r_returns = robjects.r.matrix(r_returns, nrow=nrow, ncol=ncol, byrow=robjects.vectors.BoolVector([True]))
    y = robjects.r.rep(1, nrow)
    beta0 = mte.LAD(y, r_returns)
    weights = mte.huber_lasso(y, r_returns, beta0, Lambda, c, robjects.vectors.BoolVector([False])).rx2('beta')
    boolList = list(map(lambda x: True if x != 0 else False, list(weights)))
    permnos = [i for (i, b) in zip(df.columns, boolList) if b]
    return permnos, pd.DataFrame(np.array(weights).reshape(1,-1), columns = df.columns)[permnos]

def LADLasso(df, Lambda):
    nrow = df.shape[0]
    ncol = df.shape[1]
    returns = df.values
    returns = [i for j in returns for i in j]
    r_returns = robjects.FloatVector(returns)
    r_returns = robjects.r.matrix(r_returns, nrow=nrow, ncol=ncol, byrow=robjects.vectors.BoolVector([True]))
    y = robjects.r.rep(1, nrow)
    beta0 = mte.LAD(y, r_returns)
    weights = mte.LADlasso(y, r_returns, beta0, Lambda, robjects.vectors.BoolVector([False])).rx2('beta')
    boolList = list(map(lambda x: True if x != 0 else False, list(weights)))
    permnos = [i for (i, b) in zip(df.columns, boolList) if b]
    return permnos, pd.DataFrame(np.array(weights).reshape(1,-1), columns=df.columns)[permnos]

def HuberLassoObjective(trial:Trial, returns_train=None, returns_val=None):
    Lambda_range = trial.suggest_float('Lambda', 1e-3, 1e-1)
    c_range = trial.suggest_float('c', 0.1, 1)
    permnos, weights = HuberLasso(returns_train, Lambda_range, c_range)
    ret_p = returns_val.loc[:, permnos] @ weights.values.reshape(-1, 1)
    if len(permnos) > 5:
        return calmarRatio(ret_p)
    else:
        return calmarRatio(ret_p)-1000000000

def HuberLassoOptimize(objective, returns_train, returns_val):
    study = optuna.create_study(direction='maximize')
    study.optimize(partial(objective, returns_train=returns_train, \
                           returns_val=returns_val),n_trials=200,n_jobs = -1)
    params = study.best_params
    return params

def LADLassoObjective(trial:Trial, returns_train=None, returns_val=None):
    Lambda_range = trial.suggest_float('Lambda', 1e-3, 1e-1)
    permnos, weights = LADLasso(returns_train, Lambda_range)
    ret_p = returns_val.loc[:, permnos] @ weights.values.reshape(-1,1)
    if len(permnos) > 5:
        return calmarRatio(ret_p)
    else:
        return calmarRatio(ret_p)-1000000000

def LADLassoOptimize(objective, returns_train, returns_val):
    study = optuna.create_study(direction = 'maximize')
    study.optimize(partial(objective, returns_train=returns_train, \
                   returns_val = returns_val), n_trials=200, n_jobs=-1)
    params = study.best_params
    return params

def calmarRatio(returns):
    cumlative = (returns+1).cumprod()
    mdd = abs(min(cumlative / cumlative.cummax() - 1))
    return returns.mean() / mdd
