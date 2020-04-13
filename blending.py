from collections import defaultdict
import os
import json


def blendLists(lists, k = 20, missed_rank = 0):
    """
    k=20: infinity rank
    if item is not in list -> it's rank is set to k
    ranks are summed and sorted by it
    only top k returned
    """
    key_set = set()
    store = []
    mix = defaultdict(lambda: 0)
    result = []
    for l in lists:
        s1 = defaultdict(lambda: missed_rank)
        for rank,j in enumerate(l):
            s1[j] = rank
            key_set.add(j)
        store.append(s1) 
    for i in key_set:
        for n in range(len(store)):
            mix[i] += store[n][i]
    for key, value in sorted(mix.items(), key=lambda item: item[1]): 
        result.append(key)
    return result[:k]

def blendLists2(lists, k = 20):
    """
    k=20: only top k returned
    if item is not in list -> it's 1/rank is set to 0
    1/ranks are summed and sorted by  descending
    """
    key_set = set()
    store = []
    mix = defaultdict(lambda: 0)
    result = []
    for l in lists:
        s1 = defaultdict(lambda: 0)
        for rank,j in enumerate(l):
            s1[j] = 1/(rank+1)
            key_set.add(j)
        store.append(s1) 
    for i in key_set:
        for n in range(len(store)):
            mix[i] += store[n][i]
    for key, value in sorted(mix.items(), key=lambda item: item[1], reverse=True): 
        result.append(key)
    return result[:k]    

def blendResults(names, k=20, missed_rank = 0, method='rank'):
    """
    method = rank|reverse_rank
    """

    # читаем результаты предсказаний моделей
    results = []
    for name in names:
        with open(os.path.join('submits/{0}.json'.format(name)), 'r') as f:
            res = json.load(f)
            res = {int(k): v for k, v in res.items()}
            results.append(res)  
            
    users_all = set()
    users_folds = []
    for res in results:
        users = set()
        for j in res:
            users.add(j)
            users_all.add(j)
        users_folds.append(users)  
        
    n_models = len(users_folds)
    n = 0
    blended_results = {}
    for uid in users_all:
        user_items =[]
        for mn in range(n_models):
            if uid in users_folds[mn]:
                user_items.append(results[mn][uid])
        if method == 'rank':        
            user_recs = blendLists(user_items, k=k, missed_rank=missed_rank)  
        else:
            user_recs = blendLists2(user_items, k=k)    
        blended_results[uid] = user_recs    
        
        
    return blended_results    