import pandas as pd
import numpy as np
import time

import os

import matplotlib.pyplot as plt

from tqdm import tqdm

import sys
sys.path.insert(1, 'functions')
from functions_methods import *
from functions_process_data import *
from functions_tuning import *

##################################################################
##  Load data
##################################################################
data = pd.read_csv('../data.csv',sep=';',index_col=0)
data = data.reset_index(drop=True) # Reset index

##################################################################
##  Defining explanatory variables
##################################################################
response_variable = 'npl'

# Loan-specific variables
loan_specific_variables = [
    'principal',
    'pctloan',
    'installment',
    'pctinstallment',
    'monthloan',
    'ageloan',
    ]

# Demographic variables
demographic_variables = [
    'genderd',
    'marriedd',
    'dbank',
    'daccounting',
    'dbroker',
    'dinsurance',
    'dbusecon',
    'dotherfin',
    'dgm',
    'dboard',
    'downer',
    'elementd',
    'highsd',
    'colleged',
    'masterphdd',
    'istanbuldummy',
    'ankaradummy',
    'izmirdummy',
    'age',
    ]

# Macro variables
macro_variables = [
    'cpianch',
    'uneprc',
    'holoanch',
    'fxrate',
    'intrate',
    'gdp',
    'gdppercap',
    'consconfindex',
    'constind',
    'chgmar',
    ]

# All variables
explanatory_variables_all = loan_specific_variables + demographic_variables + macro_variables

print('Number of mortgages: {}'.format(data.shape[0]))
print('Number of mortgages categorized as default: {}'.format(data[response_variable].sum()))

##################################################################
##  Handling missing values
##################################################################
print('Mortgages with missing values before imputing: {}%'.format(np.round(100*np.max(data.isnull().sum()/data.shape[0]),2)))

# variables used in the vector space for the kNN algorithm for imputation 
var_use = [
    'principal',
    'pctloan',
    'installment',
    'pctinstallment',
    'monthloan',
    'ageloan',
    'istanbuldummy',
    'ankaradummy',
    'izmirdummy',
    'cpianch',
    'uneprc',
    'holoanch',
    'fxrate',
    'intrate',
    'gdp',
    'gdppercap',
    'consconfindex',
    'constind',
    'chgmar',
 ]

data = imputing_missing_values(data,var_use)
print('Done imputing missing values')

##################################################################
##  Defining samples
##################################################################
samples = {}
samples['sampled']  = sample_rrw(data,response_variable)
samples['all']      = data

##################################################################
##  Defining test years
##################################################################
test_years = [2008,2009,2010]

##################################################################
##  Make descriptive statistics
##################################################################
# Making folder for saving descriptives
folder_descriptives = 'descriptives'
make_folder(folder_descriptives)

# Descriptives
for sample in samples:
    descriptives = pd.DataFrame(index=explanatory_variables_all)
    for i in explanatory_variables_all:
        descriptives.at[i,'Mean'] = samples[sample][i].mean()
        descriptives.at[i,'Median'] = samples[sample][i].median()
        descriptives.at[i,'Std'] = samples[sample][i].std()
    descriptives.to_excel(folder_descriptives+'/descriptives_{}.xlsx'.format(sample))

# Number of observations
sample_table_all_list = []
for sample in samples:
    unique_years = list(np.sort(samples[sample].yearloan.unique()))
    sample_table_list = []
    for yearloan in unique_years+['Total']:
        if yearloan == 'Total':
            temp = samples[sample]
        else:
            temp = samples[sample][samples[sample]['yearloan']==yearloan]
        year_series = pd.DataFrame({
            'Mortgages': [temp.shape[0]],
            'Default mortgages': [temp[response_variable].sum(axis=0)],
            'Default frequency': [temp[response_variable].sum(axis=0) / temp.shape[0]]
        }, index=[yearloan]).T
        year_series['Sample'] = sample
        year_series['Title'] = year_series.index
        year_series.set_index(['Sample', 'Title'], inplace=True)
        sample_table_list.append(year_series)
    sample_table_all_list.append(pd.concat(sample_table_list, axis=1))
sample_table_all = pd.concat(sample_table_all_list, axis=0)
sample_table_all.to_excel(folder_descriptives+'/Number of observations.xlsx')


##################################################################
## Tuning
##################################################################
method_versions_to_tune =[
    'DT',
    'RF',
    'CatBoost',
    'XGBoost',
    'LightGBM',
    'LR', # This is for finding the optimal lambda value for LASSO 
]

# Making folder for saving best_hyper_parameters_dict_all
folder_name_tuned_hyperparameters = 'tuned_hyperparameters'
make_folder(folder_name_tuned_hyperparameters)

files = os.listdir(folder_name_tuned_hyperparameters)
for sample in samples:
    if sample+'.csv' in files:
        print('File with tuned parameters existing for sample \'{}\', skipping tuning'.format(sample))
    else:
        t_total = time.time()
        best_parameters_tuned_sample = pd.DataFrame()

        for year in tqdm(test_years):
            data_train  = samples[sample][samples[sample]['yearloan']<year]

            y_train = data_train[response_variable]

            X_train = data_train[explanatory_variables_all]

            best_parameters_year = pd.DataFrame()
            for method_version in method_versions_to_tune:  
                best_parameters = tuning_rrw(method_version,X_train,y_train)
                best_parameters_year = pd.concat([best_parameters_year,pd.Series(best_parameters,name=method_version)],axis=1)

            best_parameters_year['Parameter'] = best_parameters_year.index
            best_parameters_year['Year'] = pd.Series([year]*best_parameters_year.shape[0],index = best_parameters_year.index)
            best_parameters_year.set_index(['Year', 'Parameter'], inplace=True)

            best_parameters_tuned_sample = pd.concat([best_parameters_tuned_sample,best_parameters_year],axis=0)                                        

        print('Elapset time tuning sample \'{}\': {} minutes'.format(sample,np.round(((time.time() - t_total))/60,2)))

        best_parameters_tuned_sample.to_csv(folder_name_tuned_hyperparameters+'/'+sample+'.csv',sep=';')


##################################################################
##  Analysis
##################################################################
# Get evaluation metrics
evaluation_metrics =[
    'Accuracy ratio',
    'Brier score',
    ]

method_versions =[
    'LR',
    'DT',
    'RF',
    'CatBoost',
    'XGBoost',
    'LightGBM',
]

folder_name = 'results'
make_folder(folder_name)


for sample in samples:
    # Making empty data frames for inserting results
    TOTAL_results_in_sample         = pd.DataFrame()
    TOTAL_results_out_of_sample     = pd.DataFrame()

    best_parameters_sample = pd.read_csv(folder_name_tuned_hyperparameters+'/'+sample+'.csv',sep=';',index_col=[0,1])

    columns = []
    for j in test_years:
        for i in method_versions[1:]:
            columns.append(i+'-'+str(j))
    variable_importance_table   = pd.DataFrame([[None]*len(columns)]*len(explanatory_variables_all),index=explanatory_variables_all,columns=columns)

    t_total = time.time()
    for year in test_years:

        results_in_sample           = pd.DataFrame(index=[year]+evaluation_metrics)
        results_out_of_sample       = pd.DataFrame(index=[year]+evaluation_metrics)

        # Test data shall allways be all data
        data_test   = samples['all'][samples['all']['yearloan']==year]
        data_train  = samples[sample][samples[sample]['yearloan']<year]

        for method_version in tqdm(method_versions):

            series_in_sample,series_out_of_sample,variable_importance_table = model_and_get_results(method_version,data_train,data_test,explanatory_variables_all,response_variable,year,best_parameters_sample,variable_importance_table,sample)

            # Inserting results into DataFrames
            results_in_sample[method_version] = series_in_sample
            results_out_of_sample[method_version] = series_out_of_sample

        TOTAL_results_in_sample         = pd.concat([TOTAL_results_in_sample,results_in_sample])
        TOTAL_results_out_of_sample     = pd.concat([TOTAL_results_out_of_sample,results_out_of_sample])

    print('Elapset time sample \'{}\': {} minutes'.format(sample,np.round(((time.time() - t_total))/60,2)))

    # Saving results
    TOTAL_results_in_sample.to_excel(folder_name+'/in-sample-fit - '+sample+'.xlsx',index=True)
    TOTAL_results_out_of_sample.to_excel(folder_name+'/out-of-sample - '+sample+'.xlsx',index=True)
    variable_importance_table.to_excel(folder_name+'/variable_importance_table -'+sample+'.xlsx',index=True)

