#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:54:00 2020

@author: u18277461
"""

from implicit.als import AlternatingLeastSquares
from collections import defaultdict
import scipy.sparse as sp
import numpy as np
import tqdm
import os
import metric_utils as mu

def tuneALS(tr_train, tr_val, factors=[20], cashed_model = None, t_users=[]):
    
    # test_users
    test_users = set(tr_val['user_uid'].unique())
    print('Test users done.')
    
    # filter_elements
    filtered_elements = defaultdict(set)
    
    for user_uid, element_uid in tr_train.loc[:, ['user_uid', 'element_uid']].values:
        if user_uid not in test_users:
            continue
        filtered_elements[user_uid].add(element_uid)
    print('Elements filtered out from validation')
    
    # sparse matrix
    tr_train['user_uid'] = tr_train['user_uid'].astype('category')
    tr_train['element_uid'] = tr_train['element_uid'].astype('category')

    t_matrix = sp.coo_matrix(
        (tr_train['y'].astype(np.float32) + 1,
            (
                tr_train['element_uid'].cat.codes.copy(),
                tr_train['user_uid'].cat.codes.copy()
            )
        )
    )

    t_matrix = t_matrix.tocsr()
    sparsity = t_matrix.nnz / (t_matrix.shape[0] * t_matrix.shape[1])
    print('Sparsity: %.6f' % sparsity)    
    
    #
    # iteration over parameters
    best_score = 0.0
    
    for i, factor in enumerate(factors):
        print('Iteration {0} of {1}'.format(i+1, len(factors)))
        os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
        if str(type(cashed_model))=="<class 'implicit.als.AlternatingLeastSquares'>":
            model = cashed_model
            print('Used cashed model.')
        else:    
            model = AlternatingLeastSquares(factors=factor, iterations=100)
            model.fit(item_users=t_matrix)
            print('Model {0} fitted.'.format(i+1))

        t_matrix_T = t_matrix.T.tocsr()

        t_user_uid_to_cat = dict(zip(
            tr_train['user_uid'].cat.categories,
            range(len(tr_train['user_uid'].cat.categories))
        ))
        t_element_uid_to_cat = dict(zip(
            tr_train['element_uid'].cat.categories,
            range(len(tr_train['element_uid'].cat.categories))
        ))

        t_filtered_elements_cat = {k: [t_element_uid_to_cat.get(x, None) for x in v] for k, v in filtered_elements.items()}


        # predict results
        t_result = {}

        print('Started predict/score ...')
        napks = [] #list for nap@k

        for user_uid in test_users:
            # transform user_uid to model's internal user category
            try:
                user_cat = t_user_uid_to_cat[user_uid]
            except LookupError:
                continue

            # perform inference
            t_recs = model.recommend(
                user_cat,
                t_matrix_T,
                N=20,
                filter_already_liked_items=True,
                filter_items=t_filtered_elements_cat.get(user_uid, set())
            )

            # drop scores and transform model's internal elelemnt category to element_uid for every prediction
            # also convert np.uint64 to int so it could be json serialized later
            t_result[user_uid] = [int(tr_train['element_uid'].cat.categories[i]) for i, _ in t_recs]
            napks.append(mu.napk(tr_val[tr_val['user_uid']==user_uid]['element_uid'].to_list(), t_result[user_uid], k=20))
        # score results
        print('Predict done.')
        score = np.mean(napks)
        print('MNAP@20:', score)
        if score >= best_score:
            best_score = score
            best_model = model
            print('Best model updated')
    
    #if test_users -> make prediction  
    test_result = {}
    if len(t_users) > 0:
        for user_uid in t_users:
            # transform user_uid to model's internal user category
            try:
                user_cat = t_user_uid_to_cat[user_uid]
            except LookupError:
                continue

            # perform inference
            t_recs = best_model.recommend(
                user_cat,
                t_matrix_T,
                N=20,
                filter_already_liked_items=True,
                filter_items=t_filtered_elements_cat.get(user_uid, set())
            )

            # drop scores and transform model's internal elelemnt category to element_uid for every prediction
            # also convert np.uint64 to int so it could be json serialized later
            test_result[user_uid] = [int(tr_train['element_uid'].cat.categories[i]) for i, _ in t_recs]
            
    return best_model, best_score, test_result