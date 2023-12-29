import pandas as pd
import numpy as np
import time

from sklearn import metrics
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import l1_min_c
from sklearn.linear_model import lasso_path
from sklearn import tree
# © 2007 - 2019, scikit-learn developers (BSD License).

import lightgbm as lgb

import xgboost

from catboost import CatBoostClassifier

import statsmodels.api as sm

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import os

import shap

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def get_trained_model(method_version,X_train,y_train,best_parameters_sample,year):
    
    X_train=X_train.reset_index(drop=True)
    y_train=y_train.reset_index(drop=True)

    best_parameters = best_parameters_sample[method_version][year]

    ##################################################################
    if method_version=='LR':
        X_train_standardized = pd.DataFrame(StandardScaler().fit_transform(X_train),columns=X_train.columns)

        model = linear_model.LogisticRegression(
            C       = float(best_parameters['C']),
            solver  = best_parameters['solver'],
            penalty='l1',
            max_iter=int(1e6),
            fit_intercept=True,
            random_state=0,
            )
        model.fit(X_train_standardized, y_train)

    ##################################################################
    elif method_version=='DT':
        if pd.isnull(best_parameters['max_depth']):
            max_depth = None
        else:
            max_depth = int(best_parameters['max_depth'])
        model = tree.DecisionTreeClassifier(\
            criterion           = best_parameters['criterion'],
            splitter            = best_parameters['splitter'],
            max_depth           = max_depth,
            min_samples_split   = int(best_parameters['min_samples_split']),
            min_samples_leaf    = int(best_parameters['min_samples_leaf']),
            random_state=0,
            max_leaf_nodes=None,
            max_features=None)
        model.fit(X_train, y_train)

    ##################################################################
    elif method_version=='RF':
        if pd.isnull(best_parameters['max_depth']):
            max_depth = None
        else:
            max_depth = int(best_parameters['max_depth'])
        model = RandomForestClassifier(\
            n_estimators = int(best_parameters['n_estimators']),
            max_depth = max_depth,
            random_state=0,
            warm_start=False,
            bootstrap=True,
            criterion='gini',
            max_features='sqrt')
        model.fit(X_train, y_train)

    ##################################################################
    elif method_version=='CatBoost':
        model = CatBoostClassifier(\
            iterations = int(best_parameters['iterations']),
            learning_rate = best_parameters['learning_rate'],
            depth = int(best_parameters['depth']),
            l2_leaf_reg = int(best_parameters['l2_leaf_reg']),
            verbose=False,
            random_seed=0)
        model.fit(X_train, y_train)

    ##################################################################
    elif method_version=='XGBoost':
        model = xgboost.XGBClassifier(
            learning_rate       = best_parameters['learning_rate'],
            n_estimators        = int(best_parameters['n_estimators']),
            max_depth           = int(best_parameters['max_depth']),
            subsample           = best_parameters['subsample'],
            colsample_bytree    = best_parameters['colsample_bytree'],
            min_child_weight    = int(best_parameters['min_child_weight']),
            gamma=0,
            objective= 'binary:logistic',
            random_state=1,
            )
        model.fit(X_train,y_train,eval_metric='auc')

    ##################################################################
    elif method_version=='LightGBM':        
        model = lgb.LGBMClassifier(
            learning_rate       = best_parameters['learning_rate'],
            n_estimators        = int(best_parameters['n_estimators']),
            max_depth           = int(best_parameters['max_depth']),
            subsample           = best_parameters['subsample'],
            colsample_bytree    = best_parameters['colsample_bytree'],
            min_child_weight    = int(best_parameters['min_child_weight']),
            random_state=0)

        model.fit(X_train,y_train,eval_metric='auc')

    return model


def model_and_get_results(method_version,data_train,data_test,explanatory_variables_all,response_variable,year,best_parameters_sample,variable_importance_table,sample):

    # Make training and test sets
    y_train     = data_train[response_variable].astype(int).reset_index(drop=True)
    y_test      = data_test[response_variable].astype(int).reset_index(drop=True)

    X_train     = data_train[explanatory_variables_all].astype(float).reset_index(drop=True)
    X_test      = data_test[explanatory_variables_all].astype(float).reset_index(drop=True)

    # SHAP and LASSO path
    variable_importance_table   = SHAP_and_LASSO_path_rrw(method_version,X_train,y_train,year,best_parameters_sample,variable_importance_table,sample)       

    # Train model
    model   = get_trained_model(method_version,X_train,y_train,best_parameters_sample,year)
    
    if method_version=='LR':
        # Picking only variables selected, and fitting LR model

        X_train = X_train.loc[:,(model.coef_!=0).ravel()]
        X_test  = X_test.loc[:,(model.coef_!=0).ravel()]

        if (X_train.shape[1]==0)&(X_test.shape[1]==0):
            X_train = pd.DataFrame([1]*X_train.shape[0],index=X_train.index,columns=['intercept'])
            X_test = pd.DataFrame([1]*X_test.shape[0],index=X_test.index,columns=['intercept'])

            model = linear_model.LogisticRegression(
                penalty=None,
                max_iter=int(1e6),
                fit_intercept=False,
                solver='lbfgs',
                random_state=0,
                )
            model.fit(X_train,y_train)

        else:
            model = linear_model.LogisticRegression(
                penalty=None,
                max_iter=int(1e6),
                fit_intercept=True,
                solver='lbfgs',
                random_state=0,
                )
            model.fit(X_train,y_train)

    # Results when training model with all data
    series_in_sample,series_out_of_sample   = get_metric_values(X_train,X_test,y_train,y_test,model)

    return series_in_sample,series_out_of_sample,variable_importance_table

def get_metric_values(X_train,X_test,y_train,y_test,model):
    series_in_sample        = pd.Series(dtype=float)
    series_out_of_sample    = pd.Series(dtype=float)
    
    AUC_in_sample     = metrics.roc_auc_score(y_train,model.predict_proba(X_train)[:,1])
    AUC_out_of_sample = metrics.roc_auc_score(y_test,model.predict_proba(X_test)[:,1])
    
    series_in_sample['Accuracy ratio']     = (AUC_in_sample-0.5)*2
    series_out_of_sample['Accuracy ratio'] = (AUC_out_of_sample-0.5)*2

    series_in_sample['Brier score'] = metrics.brier_score_loss(y_train,model.predict_proba(X_train)[:,1])
    series_out_of_sample['Brier score'] = metrics.brier_score_loss(y_test,model.predict_proba(X_test)[:,1])

    return series_in_sample,series_out_of_sample

def SHAP_and_LASSO_path_rrw(method_version,X_train,y_train,year,best_parameters_sample,variable_importance_table,sample):

    # Making folder for saving plots
    folder_name = 'results_plots'
    make_folder(folder_name)

    if method_version=='LR':
        # Training model
        model = get_trained_model(method_version,X_train,y_train,best_parameters_sample,year)
        # Making lasso path
        LASSO_path_rrw(model,X_train,y_train,folder_name,year,sample)

    else:
        # Training model
        model = get_trained_model(method_version,X_train,y_train,best_parameters_sample,year)

        shap_values = shap.TreeExplainer(model,feature_perturbation='tree_path_dependent').shap_values(X_train)
        if (method_version=='LightGBM')|(method_version=='RF'):
            shap_values = shap_values[1]

        SHAP_max_display = 20
        shap.summary_plot(
            shap_values, X_train,
            max_display=SHAP_max_display,
            # plot_size=None, 
            show=False
        )
        make_folder(folder_name+'/SHAP_plots_'+sample)
        plt.savefig(folder_name+'/SHAP_plots_'+sample+'/SHAP_'+method_version+'_'+str(year)+'.png',dpi=150, bbox_inches='tight')
        plt.close() # Close so the figures do not overlap

        if (method_version=='DT'):
            vals= np.abs(shap_values[0]).mean(0)
        else:
            vals= np.abs(shap_values).mean(0)

        variable_importance_table[method_version+'-'+str(year)] = 100*vals/np.max(vals)

    return variable_importance_table 


def LASSO_path_rrw(model,X_train,y_train,folder_name,year,sample):
    LASSO_lambda = 1/model.C # The lambda chosen for the LASSO method

    selected_variables = X_train.loc[:,(model.coef_!=0).ravel()].columns
    X_train_standardized = pd.DataFrame(StandardScaler().fit_transform(X_train),columns=X_train.columns)
    y_to_use = y_train.copy()

    # Making LASSO path and AUC-values
    lambda_max   = 25
    lambda_max   = 1/l1_min_c(X_train_standardized,y_to_use,loss='log',fit_intercept=False,intercept_scaling=1)
    lambda_min   = LASSO_lambda*0.95
    number_of_lambdas = 30
    lambdas_for_LASSO_path = np.linspace(lambda_max, lambda_min, number_of_lambdas)
    auc_train = []
    df_coefs = pd.DataFrame(index=X_train_standardized.columns)
    for lam in lambdas_for_LASSO_path:
        model.set_params(C=(1/lam))
        model.fit(X_train_standardized, y_to_use)
        df_coefs[lam] = pd.Series(model.coef_.ravel(),name=lam,dtype=float,index=X_train_standardized.columns)
        auc_train.append(metrics.roc_auc_score(y_to_use,model.predict_proba(X_train_standardized)[:,1]))

    # Transposing dataframe
    df_coefs = df_coefs.transpose()

    # Showing only selected variables, 
    # i.e., those non-zero at selected lambda
    df_coefs = df_coefs.loc[:,selected_variables]

    # Sort based on what variable becomes non-zero
    df_coefs = df_coefs[df_coefs.ne(0).idxmax().sort_values(ascending=False).index]

    # Plotting parameters
    fig_width   = 10 # Width of the figure
    fig_length  = 10 # Length of the figure
    linewidth   = 4  # Width of the lines in the plots
    fontsize    = 32

    # Set to True to show variable names next to lines on LASSO plot
    do_show_labels = False

    # Set to True to show a plot of the evolution of AUC on
    # the training set across different values of lambda
    do_show_plot_of_auc_train = True

    # Plot LASSO path
    fig, ax = plt.subplots(1, 1, figsize=(fig_width,fig_length))
    ax.set_prop_cycle('color', plt.cm.hsv(np.linspace(0,1,df_coefs.shape[1]))) # Defining colormap for plots
    ax.plot(df_coefs, linewidth=linewidth,label=df_coefs.columns)
    ax.set_xlim(ax.get_xlim()[::-1]) # Making descending x-axis
    if do_show_labels:
        textno=0
        y_values=[]
        for line in ax.lines:
            y = line.get_ydata()[-1]
            y_values.append(y)
            ax.annotate(df_coefs.columns[textno], xy=(.95,y), xytext=(3,0),xycoords = ax.get_yaxis_transform(), textcoords="offset points",size=fontsize, va="center")
            textno+=1
    ax.axvline(LASSO_lambda,linestyle='--',label='Chosen '+r'$\lambda$',color='navy')
    ax.legend(loc = 'upper left',bbox_to_anchor=(0.975,1.03),fontsize=fontsize,ncol=1)
    ax.set_xlabel(r'$\lambda$',fontsize=fontsize)
    ax.set_ylabel('Standardized coefficients',fontsize=fontsize)
    ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    plt.savefig(folder_name+'/LASSO_path_'+sample+'_'+str(year)+'.png',dpi=150, bbox_inches='tight')
    plt.close() # Close so the figures do not overlap

    # Plot AUC scores for LASSO path
    if do_show_plot_of_auc_train:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width,fig_length))
        ax.plot(lambdas_for_LASSO_path, auc_train, linewidth=linewidth)
        ax.axvline(LASSO_lambda,linestyle='--',label='Chosen lambda')
        ax.set_xlim(ax.get_xlim()[::-1]) # Making descending x-axis
        ax.set_xlabel(r'$\lambda$',fontsize=fontsize)
        ax.set_ylabel('AUC on training set',fontsize=fontsize)
        ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
        plt.savefig(folder_name+'/LASSO_path_AUC_scores'+'_'+str(year)+'.png',dpi=150, bbox_inches='tight')
        plt.close() # Close so the figures do not overlap
