import pandas as pd
import numpy as np

import time

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def KNNImputer_rrw(var_mis_con,var_mis_cat,var_use,data_per_yearloan):
    # All variables that are imputed are treated as continious variables
    
    # var_mis_con: CONTINIOUS variables with missing values to be imputed
    # var_mis_cat: CATEGORICAL variables with missing values to be imputed
    # var_use: variables used in the vector space for the kNN algorithm for imputation 
    
    # Making sure that index is reset.
    # If not, the process will fail
    data_per_yearloan = data_per_yearloan.reset_index(drop=True) # Reset index

    # Checking that variables used for imputation do not have missing values
    temp = data_per_yearloan[var_use].isnull().sum(axis=0)
    temp = temp[temp!=0].index.tolist()
    if len(temp)>=1:
        for i in temp:
            print('-------------------------')
            print('ERROR: Variable \'{}\' is used for k-NN imputation. However, it has missing values. Missing values are set to zero in the k-NN imputation, so consider not including this variable.'.format(i))
        print('-------------------------')
        print('Imputing failed. No values imputed.')
        print('-------------------------')

    elif len(temp)==0:
        # Imputing variable, one at a time

        # Creating vector space used for the kNN-algorithm
        data_var_use = data_per_yearloan[var_use]

        # Standardizing the vector space to a mean of 0 and 
        # standard deviation of 1 for each variable, respectively
        data_var_use = pd.DataFrame(StandardScaler().fit_transform(data_var_use),columns=data_var_use.columns)

        # First, continious variables
        for i in var_mis_con:
            # Making data such that first column is the one to be imputed
            data_for_imputation = pd.concat([data_per_yearloan[i],data_var_use],axis=1)
            num_missing_before = pd.isnull(data_per_yearloan[i]).sum(axis=0)

            # Imputing
            imputer = KNNImputer(n_neighbors=3,weights='distance')
            temp = pd.DataFrame(imputer.fit_transform(data_for_imputation))

            # First column is the one imputed, so replacing this one in data
            data_per_yearloan[i] = temp[0]

            num_missing_after = pd.isnull(data_per_yearloan[i]).sum(axis=0)
            # print('Imputed {} instances of missing values for continious variable \'{}\''.format(num_missing_before-num_missing_after,i))
            # print('Elapset time: {} minutes'.format(np.round((time.time()-t)/60,2)))
            if num_missing_after!=0:
                print('ERROR: Imputation failed: still {} instances of missing values for variable {}'.format(num_missing_after,i))

        # Second, categorical variables including dummies
        for i in var_mis_cat:
            # Making data such that first column is the one to be imputed
            data_for_imputation = pd.concat([data_per_yearloan[i],data_var_use],axis=1)
            num_missing_before = pd.isnull(data_per_yearloan[i]).sum(axis=0)
            num_cat_values_before = len(data_per_yearloan[i][pd.isnull(data_per_yearloan[i])==False].unique())
            
            # Imputing
            imputer = KNNImputer(n_neighbors=1)
            temp = pd.DataFrame(imputer.fit_transform(data_for_imputation))

            # First column is the one imputed, so replacing this one in data
            data_per_yearloan[i] = temp[0]

            num_missing_after = pd.isnull(data_per_yearloan[i]).sum(axis=0)
            num_cat_values_after = len(data_per_yearloan[i][pd.isnull(data_per_yearloan[i])==False].unique())

            if num_cat_values_before!=num_cat_values_after:
                print('ERROR: number of categorical values before and after impotation are {} and {}, respectively.'.format(num_cat_values_before,num_cat_values_after))
            if num_missing_after!=0:
                print('ERROR: Imputation failed: still {} instances of missing values for variable {}'.format(num_missing_after,i))

    return data_per_yearloan

def sample_rrw(data,response_variable):

    unique_years = data['yearloan'].unique()

    # Shuffle entire DataFrame and reset index
    data = data.sample(frac=1,random_state=0).reset_index(drop=True)

    list_true   = data[data[response_variable]==1].index.tolist()

    data_false = data[data[response_variable]==0]

    list_false = []
    for i in range(len(unique_years)):
        temp = data_false[data_false['yearloan']==unique_years[i]].index.tolist()
        list_false = list_false+temp[0:int(np.round(len(temp)*0.01))]
    
    if len(np.unique(list_false))!=len(list_false):
        print('ERROR in sample(): not all in list_false are unique')

    data_to_use = data.loc[list_true+list_false]
    data_to_use = data_to_use.reset_index(drop=True) # Reset index

    return data_to_use


def imputing_missing_values(data,var_use):
    if False:
        # Show missing values
        temp_data = data[data.yearloan<=2009]
        temp = temp_data.isnull().sum(axis=0)
        temp[temp!=0]/temp_data.shape[0]

        # Per year:
        pd.pivot_table(\
            pd.concat([pd.isnull(data),pd.Series(data['yearloan'],name='aar')],axis=1),
            values='genderd',
            index=['aar'],
            aggfunc='sum')
        # Fraction per year
        pd.pivot_table(\
            pd.concat([pd.isnull(data),pd.Series(data['yearloan'],name='aar')],axis=1),
            values='genderd',
            index=['aar'],
            # columns=['C'],
            aggfunc=np.mean)

    # CONTINIOUS variables with missing values to be imputed
    var_mis_con = [
        'age',
    ]

    # CATEGORICAL variables with missing values to be imputed
    var_mis_cat = [
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
    ]

    num_obs_before = data.shape[0]
    check_sum_before = data['customerno'].sum(axis=0)

    data_without_missing = pd.DataFrame(columns=data.columns)
    unique_yearloan = np.sort(data['yearloan'].unique())
    for yearloan in unique_yearloan:
        data_per_yearloan = data[data['yearloan']==yearloan]
        data_per_yearloan = KNNImputer_rrw(var_mis_con,var_mis_cat,var_use,data_per_yearloan)
        if yearloan == unique_yearloan[0]:
            data_without_missing = data_per_yearloan.copy()
        else:
            data_without_missing = pd.concat([data_without_missing,data_per_yearloan])

    data_without_missing = data_without_missing.reset_index(drop=True)
    
    if num_obs_before != data_without_missing.shape[0]:
        print('ERROR: handling missing values failed')
    if check_sum_before != data_without_missing['customerno'].sum(axis=0):
        print('ERROR: handling missing values failed')

    if pd.isnull(data_without_missing).sum(axis=0).sum(axis=0)!=0:
        print('ERROR: There are still missing values in the data.')

    return data_without_missing
