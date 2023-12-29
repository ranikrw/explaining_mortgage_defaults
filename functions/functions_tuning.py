import pandas as pd
import numpy as np

import xgboost

import os
import time
import pickle

from sklearn.svm import l1_min_c
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
# © 2007 - 2019, scikit-learn developers (BSD License).

from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier

import lightgbm as lgb


def tuning_rrw(method_version,X_train,y_train):
    
    y_train = y_train.astype(int)
    X_train = X_train.astype(float)

    ##################################################################
    if method_version=='LR':
        X_train = pd.DataFrame(StandardScaler().fit_transform(X_train),columns=X_train.columns)
        
        num = 20
        start = 0
        stop = 7
        cs = l1_min_c(X_train, y_train, loss='log')*np.logspace(start, stop, num)
        
        param_grid = [{\
            'C': cs,
            'solver': ['saga','liblinear'],
            }]
        model = linear_model.LogisticRegression(\
            penalty='l1',
            max_iter=int(1e6),
            fit_intercept=True,
            random_state=0)
        grid_search = GridSearchCV(
            model,
            n_jobs = -1,
            param_grid = param_grid,
            scoring='roc_auc',
            refit=True,
            cv=3).fit(X_train, y_train)
        best_parameters = grid_search.best_params_

    ##################################################################
    elif method_version == 'XGBoost':
        param_grid = {
            'learning_rate': (0.1,0.2,0.3),
            'n_estimators': (50,100,150),
            'max_depth': (3,5,7),
            'subsample':  (0.5,0.75,1), 
            'colsample_bytree': (0.5,0.75,1),
            'min_child_weight':(1,3,5)
        }
        model = xgboost.XGBClassifier(
                gamma=0,
                objective= 'binary:logistic',
                random_state=1,
                eval_metric='auc',
                )
        grid_search = GridSearchCV(
            model,
            n_jobs = -1,
            param_grid = param_grid,
            scoring='roc_auc',
            refit=True,
            cv=3).fit(X_train,y_train)
        best_parameters = grid_search.best_params_

    ##################################################################
    elif method_version=='DT':
        param_grid = [{\
            'criterion': ('gini','entropy'),
            'splitter': ('best','random'),
            'max_depth': (5,7,9,None),
            'min_samples_split': (1,2,3),
            'min_samples_leaf': (1,2,3),
            }]
        model = tree.DecisionTreeClassifier(\
            random_state=0,
            max_leaf_nodes=None,
            max_features=None)
        grid_search = GridSearchCV(
            model,
            n_jobs = -1,
            param_grid = param_grid,
            scoring='roc_auc',
            refit=True,
            cv=3).fit(X_train, y_train)
        best_parameters = grid_search.best_params_

    ##################################################################
    elif method_version=='RF':
        param_grid = [{\
            'criterion': ('gini','entropy'),
            'n_estimators': (50,100,150),
            'max_depth': (5,7,9,None),
            }]
        model = RandomForestClassifier(\
            random_state=0,
            warm_start=False,
            bootstrap=True,
            max_features=None)
        grid_search = GridSearchCV(
            model,
            n_jobs = -1,
            param_grid = param_grid,
            scoring='roc_auc',
            refit=True,
            cv=3).fit(X_train, y_train)
        best_parameters = grid_search.best_params_

    ##################################################################
    elif method_version=='CatBoost':
        param_grid = [{\
            'iterations': (50,100,150),
            'learning_rate': (0.1,0.2,0.3),
            'depth': (5,7,9),
            'l2_leaf_reg': (1, 3, 5, 9),
            }]
        model = CatBoostClassifier(\
            verbose=False,
            random_seed=0)
        temp = model.grid_search(param_grid,X=X_train,y=y_train,
            plot=False,
            cv=3,
            partition_random_seed=0,
            verbose=False,
            calc_cv_statistics=False)
        best_parameters = temp['params']

    ##################################################################
    elif method_version == 'LightGBM':
        param_grid = {
            'learning_rate': (0.1,0.2,0.3),
            'n_estimators': (50,100,150),
            'max_depth': (-1,3,5,7),
            'subsample':  (0.5,0.75,1), 
            'colsample_bytree': (0.5,0.75,1),
            'min_child_weight':(1e-3,1e-2,1e-1,1,10)
        }
        model = lgb.LGBMClassifier(random_state=0)
        grid_search = GridSearchCV(
            model,
            n_jobs = -1,
            param_grid = param_grid,
            scoring='roc_auc',
            refit=True,
            cv=3).fit(X_train,y_train,eval_metric='auc')
        best_parameters = grid_search.best_params_

    ##################################################################
    else:
        print('ERROR: Wrongly defined method_version')

    return best_parameters

