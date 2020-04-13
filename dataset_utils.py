#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:52:19 2020

@author: u18277461

Dataset Utils
"""
import os
import json
import pandas as pd
import numpy as np

# settings
DATA_PATH = './'

def processCatalogue():
    
    with open(os.path.join(DATA_PATH, 'catalogue.json'), 'r') as f:
        catalogue = json.load(f)
    
    catalogue = {int(k): v for k, v in catalogue.items()}
    
    element_id_lst = []
    target_time_lst = []
    can_subscribe_lst = []
    can_rent_lst = []
    can_purchase_lst = []
    duration_sec_lst = []
    f1_lst = []
    f2_lst = []
    f3_lst = []
    f4_lst = []
    f5_lst = []
    is_movie_lst = []
    is_multipart_movie_lst = []
    is_series_lst = []
    len_attr_lst = []
    a_lst = [[] for i in range(20)]
    
    for i in catalogue:
        element_id_lst.append(i)
        len_attr_lst.append(len(catalogue[i]['attributes']))
        f1_lst.append(catalogue[i]['feature_1'])
        f2_lst.append(catalogue[i]['feature_2'])
        f3_lst.append(catalogue[i]['feature_3'])
        f4_lst.append(catalogue[i]['feature_4'])
        f5_lst.append(catalogue[i]['feature_5'])
        
        avs = set(catalogue[i]['availability'])
        duration_sec = catalogue[i]['duration'] * 60
        
        duration_sec_lst.append(duration_sec)
        
        if 'purchase' in avs:
            can_purchase_lst.append(1)
        else:
            can_purchase_lst.append(0)
        if 'rent' in avs:
            can_rent_lst.append(1)
        else:
            can_rent_lst.append(0)
        if 'subscription' in avs:
            can_subscribe_lst.append(1)
            if catalogue[i]['type'] in set(['movie', 'multipart_movie']):
                target_time_lst.append(duration_sec/2)
            else:
                target_time_lst.append(duration_sec/3)
        else:
            can_subscribe_lst.append(0)    
            target_time_lst.append(0)
        
        if catalogue[i]['type'] == 'movie':
            is_movie_lst.append(1)
        else:
            is_movie_lst.append(0)
            
        if catalogue[i]['type'] == 'multipart_movie':
            is_multipart_movie_lst.append(1)
        else:
            is_multipart_movie_lst.append(0)   
            
        if catalogue[i]['type'] == 'series':
            is_series_lst.append(1)
        else:
            is_series_lst.append(0)       
            
        for a_idx in range(20):
            if a_idx < len(catalogue[i]['attributes']):
                a_lst[a_idx].append(catalogue[i]['attributes'][a_idx])
            else:    
                a_lst[a_idx].append(0)            
        
            
    ds = pd.DataFrame(
        {
            'element_id': element_id_lst,
            'target_time': target_time_lst,
            'can_subscribe': can_subscribe_lst,
            'can_rent': can_rent_lst,
            'can_purchase': can_purchase_lst,
            'duration_sec': duration_sec_lst,
            'f1': f1_lst,
            'f2': f2_lst,
            'f3': f3_lst,
            'f4': f4_lst,
            'f5': f5_lst,
            'is_movie': is_movie_lst,
            'is_multipart_movie': is_multipart_movie_lst,
            'is_series': is_series_lst,
            'len_attr_lst': len_attr_lst,
            'a0': a_lst[0],
            'a1': a_lst[1],
            'a2': a_lst[2],
            'a3': a_lst[3],
            'a4': a_lst[4],
            'a5': a_lst[5],
            'a6': a_lst[6],
            'a7': a_lst[7],
            'a8': a_lst[8],
            'a9': a_lst[9],
            'a10': a_lst[10],
            'a11': a_lst[11],
            'a12': a_lst[12],
            'a13': a_lst[13],
            'a14': a_lst[14],
            'a15': a_lst[15],
            'a16': a_lst[16],
            'a17': a_lst[17],
            'a18': a_lst[18],
            'a19': a_lst[19]
        }
    )    
    
    return ds


def processTestUsers():
    
    with open(os.path.join(DATA_PATH, 'test_users.json'), 'r') as f:
        test_users = set(json.load(f)['users'])
    
    return test_users

def dfTestUsers():
    
    test_users = processTestUsers()
    df = pd.DataFrame(
            {'user_uid': list(test_users)}
            )
    return df


def processTransactions(use_cashe=False):
    
    if use_cashe==True:
        try:
            transactions = pd.read_pickle('datasets/transactions.pkl')
        except Exception:
            return processTransactions(use_cashe=False)
    else:
        
        transactions = pd.read_csv(
        os.path.join(DATA_PATH, 'transactions.csv'),
        dtype={
            'element_uid': np.uint16,
            'user_uid': np.uint32,
            'consumption_mode': 'category',
            'ts': np.float64,
            'watched_time': np.uint64,
            'device_type': np.uint8,
            'device_manufacturer': np.uint8
            }
        )
        
        transactions.to_pickle('datasets/transactions.pkl')
        
    return transactions    
        

def processRatings():
    
    ratings = pd.read_csv(
        os.path.join(DATA_PATH, 'ratings.csv'),
        dtype={
            'element_uid': np.uint16,
            'user_uid': np.uint32,
            'ts': np.float64,
            'rating': np.uint8
        }
    )
    
    return ratings

def processBookmarks():
    
    bookmarks = pd.read_csv(
        os.path.join(DATA_PATH, 'bookmarks.csv'),
        dtype={
            'element_uid': np.uint16,
            'user_uid': np.uint32,
            'ts': np.float64
        }
    )
    
    return bookmarks


def trainvalTransactions(tr, border=0.9):
    """
    Splits tr dataset to train/val parts
    in order by ts!
    border is train share
    """
    
    min_ts = tr['ts'].min()
    max_ts = tr['ts'].max()
    
    border_ts = min_ts + (max_ts - min_ts) * border
    
    tr_train = tr[tr['ts'] <= border_ts]
    tr_val = tr[tr['ts'] > border_ts]
    
    return tr_train, tr_val

def _makePurchaseTarget(row, params):
    
    y = 0
    
    if row['consumption_mode'] in set(['P','R']):
        y = params['purchased']
        
    else:
        if row['watched_time'] >= row['target_time']:
            if row['is_series'] == 1:
                y = params['watched_series']
            else:
                y = params['watched_movie']
        else:
            y = params['watch_failed']
        
    return y        


def makeTarget(ds, catalogue, bookmarks, ratings, params = {'bookmarked': 5, 'purchased': 10, 
                             'watched_movie': 8, 'watched_series': 9, 'watch_failed': 1}, tr_weight = 1.0, blend_rating=False):
        
    # add catalogue
    features_list = ['element_uid', 'user_uid', 'watched_time', 'consumption_mode']
    ds = ds[features_list]
    
    ds = ds.merge(catalogue[['element_id', 'target_time', 'is_series']], 
                          left_on='element_uid', right_on='element_id', how = 'left')
    
    # add ratings converted to scale -5.0 -- + 5.0
    # NaN = 0
    ds = ds.merge(ratings[['user_uid', 'element_uid', 'rating']], left_on=['user_uid', 'element_uid'], right_on=['user_uid', 'element_uid'], how='left')
    ds[['rating']] = ds[['rating']].fillna(value=5.0)
    ds['rating'] = ds['rating'] - 5.0

    # add bookmarks
    # NaN = 0
    bookmarks['bm_rating'] = params['bookmarked']
    ds = ds.merge(bookmarks[['user_uid', 'element_uid', 'bm_rating']], left_on=['user_uid', 'element_uid'], right_on=['user_uid', 'element_uid'], how='left')
    ds[['bm_rating']] = ds[['bm_rating']].fillna(value=0.0)

    ds['tr_rating'] = ds.apply(lambda row: _makePurchaseTarget(row, params), axis=1)

    if blend_rating:
        # blend ratings
        ds['y'] = ds['tr_rating'] + ds['rating'] + ds['bm_rating']

        # convert to 0-10 scale
        ds['y'] = ds['y'] - ds['y'].min()
        ds['y'] = ds['y'] * (10.0/ds['y'].max())
    else:
        ds['y'] = ds['tr_rating']    

    return ds[['user_uid', 'element_uid', 'y', 'tr_rating']]

def makeTargetRating(ds):
    ds['y'] = ds['rating']
    return ds[['user_uid', 'element_uid', 'y']]

def makeTargetBookmarks(ds):
    ds['y'] = 8
    return ds[['user_uid', 'element_uid', 'y']]
        