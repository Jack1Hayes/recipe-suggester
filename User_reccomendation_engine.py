# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:51:41 2022

@author: Hayes
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

I = pd.read_csv('file path for interactions_train.csv')
R = pd.read_csv(' file path for RAW_recipes.csv')

#show recipe rating distribution

#I.rating.value_counts().plot(kind = 'bar', fontsize = 14,
#                             figsize = (5, 2)).set_title('Distribution of Rating',
#                                                         fontsize = 16, ha = 'center', va = 'bottom')




#group recipies and user IDs
grouped_1 = I.groupby(['user_id'], as_index = False, sort = False).agg({'recipe_id':'count'}).reset_index(drop = True)
grouped_1 = grouped_1.rename(columns = {'recipe_id':'reviews_count'})
grouped_1 = grouped_1.sort_values('reviews_count', ascending = False).iloc[:7500, :]

grouped_2 = I.groupby(['recipe_id'], as_index = False, sort = False).agg({'user_id':'count'}).reset_index(drop = True)
grouped_2 = grouped_2.rename(columns = {'user_id':'reviews_count'})
grouped_2 = grouped_2.sort_values('reviews_count', ascending = False).iloc[:7500, :]

_part = pd.merge(I.merge(grouped_1).drop(['reviews_count'], axis = 1), grouped_2).drop(['reviews_count'], axis = 1)

#print('unique users:',len(_part.user_id.unique()))
#print('unique recipes:',len(_part.recipe_id.unique()))





#Show stats for users and recipies 

grouped_user = _part.groupby(['user_id'], as_index = False, sort = False).agg({'recipe_id':'count'}).reset_index(drop = True)
grouped_user = grouped_user.rename(columns = {'recipe_id':'reviews_count'})

#display(grouped_user[['reviews_count']].describe())

grouped_recipe = _part.groupby(['recipe_id'], as_index = False, sort = False).agg({'user_id':'count'}).reset_index(drop = True)
grouped_recipe = grouped_recipe.rename(columns = {'user_id':'reviews_count'})

#display(grouped_recipe[['reviews_count']].describe())


#_part.rating.value_counts().plot(kind = 'bar', fontsize = 14, 
#                                 figsize = (5, 2)).set_title('Distribution of Rating',
#                                                             fontsize = 16, ha = 'center', va = 'bottom')

#plt.show()







#Map the values of rating into the train and test data sets

new_userID = dict(zip(list(_part['user_id'].unique()),
                      list(range(len(_part['user_id'].unique())))))

#display(new_userID)

new_recipeID = dict(zip(list(_part['recipe_id'].unique()),
                        list(range(len(_part['recipe_id'].unique())))))

#display(new_recipeID)

df = _part.replace({'user_id': new_userID, 'recipe_id': new_recipeID})





#print('The recipes without names: ', R['id'][R['name'].isnull()].values[0])

#display(df[df['recipe_id'] == R['id'][R['name'].isnull()].values[0]])

recipe = R[['name', 'id', 'ingredients']].merge(_part[['recipe_id']], 
                                                left_on = 'id', right_on = 'recipe_id', 
                                                how = 'right').drop(['id'], axis = 1).drop_duplicates().reset_index(drop = True)

# apply centered cosine to the “df.” minimizes the differences between the “hard raters” and “easy raters.”

mean = df.groupby(['user_id'], as_index = False, sort = False).mean().rename(columns = {'rating':'rating_mean'})
df = df.merge(mean[['user_id','rating_mean']], how = 'left')
df.insert(2, 'rating_adjusted', df['rating'] - df['rating_mean'])




# To measure the performance of this recommendation engine in the next steps, I am going to split the “df” into ¾ train data set and ¼ test data set.

train_data, test_data = train_test_split(df, test_size = 0.25)
n_users = df.user_id.unique()
n_items = df.recipe_id.unique()
train_data_matrix = np.zeros((n_users.shape[0], n_items.shape[0]))
for row in train_data.itertuples():
    train_data_matrix[row[1]-1, row[2]-1] = row[3]
#display(train_data_matrix.shape)
#display(train_data_matrix)


test_data_matrix = np.zeros((n_users.shape[0], n_items.shape[0]))
for row in test_data.itertuples():
    test_data_matrix[row[1]-1, row[2]-1] = row[3]
#display(test_data_matrix.shape)
#display(test_data_matrix)




#There are two major types of memory-based collaborative filtering.

#User-based: “Users who are similar to you also like …”

#Item-based: “Users who like this item also like …”

#Since I am not sure which type has a better prediction for this data set at this point, I am going to calculate the cosine similarity for both types
user_similarity = 1 - pairwise_distances(train_data_matrix, metric = 'cosine')
#display(user_similarity.shape)
#display(user_similarity)

item_similarity = 1 - pairwise_distances(train_data_matrix.T, metric = 'cosine')
#display(item_similarity.shape)
#display(item_similarity)


def predict(ratings, similarity, _type = 'user'):
    if _type == 'user':
        pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis = np.newaxis)])
    
    elif _type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis = 1)]) 
    
    return pred

user_pred = predict(train_data_matrix, user_similarity, _type = 'user')
#display(user_pred.shape)                       
#display(user_pred)
user_pred_df = pd.DataFrame(user_pred, columns = list(n_items))
user_pred_df.insert(0, 'user_id', list(n_users))



#For the prediction by using the item similarity, I let the matrix of the train data set dot the matrix of item similarity to get a weighted sum.

item_pred = predict(train_data_matrix, item_similarity, _type = 'item')
#display(item_pred.shape)
#display(item_pred)

item_pred_df = pd.DataFrame(item_pred, columns = list(n_items))
item_pred_df.insert(0, 'user_id', list(n_users))


#Evaluation of the predictions
'''
def RMSE(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    
    return sqrt(mean_squared_error(prediction, ground_truth))

#user_RMSE = RMSE(user_pred, test_data_matrix)
#item_RMSE = RMSE(item_pred, test_data_matrix)
#print('user_RMSE = {}'.format(user_RMSE))
#print('item_RMSE = {}'.format(item_RMSE))
'''


#recomendation engine for user

def getRecommendations_UserBased(user_id, top_n = 10):
    for old_user, new_user in new_userID.items():
        if user_id == new_user:
            print(f'Top {top_n} Recommended Recipes for Original User ID: {old_user}\n')
    
    movie_rated = list(df['recipe_id'].loc[df['user_id'] == user_id])
    _all = user_pred_df.loc[user_pred_df['user_id'] == user_id].copy()
    _all.drop(user_pred_df[movie_rated], axis = 1, inplace = True)
    unwatch_sorted = _all.iloc[:,1:].sort_values(by = _all.index[0], axis = 1, ascending = False)
    dict_top_n = unwatch_sorted.iloc[:, :top_n].to_dict(orient = 'records')

    i = 1
    for recipe_id in list(dict_top_n[0].keys()):
        for old_recipe, new_recipe in new_recipeID.items():
            if recipe_id == new_recipe:
                name = recipe[recipe['recipe_id'] == old_recipe]['name'].values[0]
                ingredients = recipe[recipe['recipe_id'] == old_recipe]['ingredients'].values[0]

                print(f'Top {i} Original Recipe ID: {old_recipe} - {name}\n Ingredients: {ingredients}\n')
                
                i += 1
                
    return dict_top_n[0]

def getRecommendations_RecipeBased(Recipe_id, top_n = 10):
    for old_recipe, new_recipe in new_recipeID.items():
        name = recipe[recipe['recipe_id'] == old_recipe]['name'].values[0]
        if Recipe_id == new_recipe:
            print(f'Top {top_n} Recommended Recipes for Recipe ID: {old_recipe} - {name}\n')
    
    movie_rated = list(df['recipe_id'].loc[df['recipe_id'] == Recipe_id])
    _all = item_pred_df.loc[item_pred_df['user_id'] == Recipe_id].copy()
    _all.drop(item_pred_df[movie_rated], axis = 1, inplace = True)
    unwatch_sorted = _all.iloc[:,1:].sort_values(by = _all.index[0], axis = 1, ascending = False)
    dict_top_n = unwatch_sorted.iloc[:, :top_n].to_dict(orient = 'records')

    i = 1
    for recipe_id in list(dict_top_n[0].keys()):
        for old_recipe, new_recipe in new_recipeID.items():
            if recipe_id == new_recipe:
                name = recipe[recipe['recipe_id'] == old_recipe]['name'].values[0]
                ingredients = recipe[recipe['recipe_id'] == old_recipe]['ingredients'].values[0]

                print(f'Top {i} Original Recipe ID: {old_recipe} - {name}\n Ingredients: {ingredients}\n')
                
                i += 1
                
    return dict_top_n[0]

#R1_UserBased = getRecommendations_UserBased(702)
#R2_UserBased = getRecommendations_RecipeBased(243)
