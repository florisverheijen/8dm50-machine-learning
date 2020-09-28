# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 20:33:24 2020

@author: Verheyen
"""
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import KFold

gene_expression = pd.read_csv("RNA_expression_curated.csv", sep=',', header=0, index_col=0)
drug_response = pd.read_csv("drug_response_curated.csv", sep=',', header=0, index_col=0)

# get values of de pandas Dataframes
X = gene_expression.values
y = drug_response.values

feature_names = gene_expression.columns

df_freq_select = pd.DataFrame(feature_names)


   

repeat = 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#loop
for i in range(repeat):
    

    # split data in train and test set
    # random state ensures that the split is reproducible, so each time you run the train and test set will be the same
    
    yr_train = np.ravel(y_train)
    X_boots = X_train[np.random.randint(0,118,1)[0]][np.random.randint(0,237,50)]
    y_boots = np.random.choice(yr_train,replace=False,size = 50)
    
    # standardize X values
    scaler = StandardScaler()
    scaler.fit(X_boots)
    X_train_s = X_boots
    
    # define parameters for Lasso regression
    lasso = Lasso(random_state = 0, max_iter = 10000)
    alphas = np.logspace(-2,1,50)
    tuned_parameters = [{'alpha':alphas}]
    n_folds = 5
    
    # perform gridsearch on all parameters alpha and fit model
    grid = GridSearchCV(lasso,tuned_parameters, cv = n_folds)
    grid.fit(X_train_s,y_boots)
    
    # get mean test score and standard deviation on all samples
    scores = grid.cv_results_['mean_test_score']    
    scores_std = grid.cv_results_['std_test_score']
    
    print("Best parameter: ", grid.best_params_)
    
    # refitting on the best parameter alpha
    best_model = grid.best_estimator_
    y_pred = best_model.fit(X_train_s,y_boots).predict(X_test) # perform fitting and predicting in one step


    # fit model with best found value for alpha, so with minimum cross-validation error
    
    clf = Lasso(alpha = list(grid.best_params_.values())).fit(X_train_s, y_boots)
    
    
    
    
    for i in range(len(feature_names)):
        if clf.coef_[i] > 0 :
           print(feature_names[i],clf.coef_[i]) 
    
    # the features with the highest coef_ are most important
    importance = np.abs(clf.coef_)
    
    # get the indices of nr_imp most important features
    nr_imp = 5
    imp_index = (-importance).argsort() # sort at descending importance, so the first ones have the highest importance
    i_most_imp_features = imp_index[:nr_imp]
    
    # get all feature names
    
    # print feature names of nr_imp features
    most_imp_features = np.array(feature_names[i_most_imp_features])
   
    # Use k-fold cross validation to check whether the obtained 'best' value for alpha is independent
    # of the train/test division, so for different subsets of the data
    lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=10000)
    nr_k = 7
    k_fold = KFold(nr_k)
    
    alpha_list = []
    data = [[] for _ in range(nr_k)] 
    nz_coef = [0] * nr_k
    for k, (train, test) in enumerate(k_fold.split(X, y)):
        yr = np.ravel(y) # reshape y from column-vector to 1d array
        lasso_cv.fit(X[train], yr[train])
        alpha_k = lasso_cv.alpha_
        alpha_list.append(alpha_k)
        #print("[fold {0}] alpha: {1:.4f}, score: {2:.4f}".
              #format(k, alpha_k, lasso_cv.score(X[test], yr[test])))
    
        clf = Lasso(alpha = alpha_k).fit(X[train], yr[train])
        importance = np.abs(clf.coef_)     #get most important features
    
        # get the indices of nr_imp most important features
        imp_index = (-importance).argsort() # sort at descending importance, so the first ones have the highest importance
        i_most_imp_features = imp_index[:nr_imp]
    
        # get feature names
        feature_names = gene_expression.columns
        most_imp_features = np.array(feature_names[i_most_imp_features])
    
        # store data for DataFrame
        data[k].append(most_imp_features)
        nz_coef[k] = np.array(np.nonzero(importance)).shape[1]
        




    



































