# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:29:30 2022

@author: Hayes


"""

import numpy as np
import pandas as pd
import random
import sys
sys.path.append(r"C:\Users\Hayes\OneDrive\Documents\Data_sets")

from User_reccomendation_engine import getRecommendations_RecipeBased as gr
from User_reccomendation_engine import I, R, grouped_1, grouped_2


part = pd.merge(I.merge(grouped_1).drop(['reviews_count'], axis = 1), grouped_2).drop(['reviews_count'], axis = 1)

mean = part.groupby(['recipe_id'], as_index = False, sort = False).mean().rename(columns = {'rating':'rating_mean'})

df = pd.merge(mean, R, left_on=['recipe_id'], 
             right_on= ['id'], how='left')

new_recipeID = dict(zip(list(part['recipe_id'].unique()),
                        list(range(len(part['recipe_id'].unique())))))

df = df.replace({'recipe_id': new_recipeID})

#display(new_recipeID)




def UserInput():

    user_input = input('input: ')
    
    
    y = str(user_input)
    
    x = user_input.replace('[', '')
    x = x.replace(']', '')
    x = x.replace("'", "")
    x = x.replace(",", "")
    x = x.split(" ")
    
    
    
    recipe_id_select = []
    recipe_id_select_2 = []
    
    
    for i, j in zip(df['ingredients'], df['id']):
        if all(item in i for item in x): 
            recipe_id_select.append(j)
    
    
    for l, m in zip(df['ingredients'], df['id']):
        l = l.replace('[', '')
        l = l.replace(']', '')
        l = l.replace("'", "")
        l = l.replace(",", "")
        l = l.split(" ")
        if all(item in y for item in l):
            recipe_id_select_2.append(m)
    
    return recipe_id_select, recipe_id_select_2
    
           
def randomrecipeFinder():
    
    if len(recipe_id_select) > 5:
        randomElement = random.choices(recipe_id_select, k=5)
        r1_table = pd.DataFrame(randomElement)
    else: 
        r1_table = pd.DataFrame(recipe_id_select)
        
    if len(recipe_id_select_2) > 5:
        randomElement1 = random.choices(recipe_id_select_2, k=5)
        r2_table = pd.DataFrame(randomElement1)
    else: 
        r2_table = pd.DataFrame(recipe_id_select_2)
            
    return r1_table, r2_table
      
def recipetables():            
    try:
        recipes_1 = pd.merge(r1_table, R, left_on=[0], 
                     right_on= ['id'], how='left')
        recipes_1 = recipes_1.drop([0, 'contributor_id', 'nutrition', 'submitted', 'n_steps', 'tags', 'n_ingredients'], axis = 1)
        print(recipes_1.drop(['steps'], axis = 1))
        recipes_1
    except:
        recipes_1 = print("We can't find any general recipies that match these ingredients. Try removing some ingredients, check the spelling or use only single ingridients without an 's' at the end")
        
    
    try:
        recipes_2 = pd.merge(r2_table, R, left_on=[0], 
                     right_on= ['id'], how='left')
        recipes_2 = recipes_2.drop([0, 'contributor_id', 'nutrition', 'submitted', 'n_steps', 'tags', 'n_ingredients'], axis = 1)
        print(recipes_2.drop(['steps'], axis = 1))
    except:
        recipes_2 = print("We can't find any specific recipies that match these ingredients. Try adding some ingredients, check the spelling or use only single ingridients without an 's' at the end")

    return recipes_1, recipes_2
    

def nextSteps():
    
    
    try:        
        id_ = (int(recipes_1.iloc[[next_step]]['id']))
        id_ = df['recipe_id'].where(df['id'] == id_)
        id_ = id_.dropna()
        id_ = id_.item()
        
        
        if next_step == '0':      
            print(recipes_1.iloc[[0]].drop(['description'], axis = 1))        
        elif next_step == '1':
            print(recipes_1.iloc[[1]].drop(['description'], axis = 1))
        elif next_step == '2':
            print(recipes_1.iloc[[2]].drop(['description'], axis = 1))
        elif next_step == '3':
            print(recipes_1.iloc[[3]].drop(['description'], axis = 1))
        elif next_step == '4':
            print(recipes_1.iloc[[4]].drop(['description'], axis = 1))
        elif next_step =='S' or 's':
            print('finished')
        
        return next_step, id_
    except:
        try:        
            id_ = (int(recipes_2.iloc[[next_step]]['id']))
            id_ = df['recipe_id'].where(df['id'] == id_)
            id_ = id_.dropna()
            id_ = id_.item()
            
            
            if next_step == '0':      
                print(recipes_2.iloc[[0]].drop(['description'], axis = 1))        
            elif next_step == '1':
                print(recipes_2.iloc[[1]].drop(['description'], axis = 1))
            elif next_step == '2':
                print(recipes_2.iloc[[2]].drop(['description'], axis = 1))
            elif next_step == '3':
                print(recipes_2.iloc[[3]].drop(['description'], axis = 1))
            elif next_step == '4':
                print(recipes_2.iloc[[4]].drop(['description'], axis = 1))

            
            return next_step, id_
        except: 
            return next_step
        return next_step
    
    
    

next_step = 'I' 

while next_step != '1' or next_step != '2' or next_step != '3' or next_step != '4' or next_step != '0' or next_step != 's' or next_step !='S':
    if  next_step == 'I' or next_step == 'i':
        recipe_id_select, recipe_id_select_2 = UserInput()
        r1_table, r2_table = randomrecipeFinder()
        recipes_1, recipes_2 = recipetables()
        next_step = input('Press 0, 1, 2, 3 or 4 to go into the recipe steps, Press R for new recipes  or Press I to change ingredients, or S to stop: ')
        try:
            next_step, id_ = nextSteps()                     
        except:
            pass
        if next_step == '1' or next_step == '2' or next_step == '3' or next_step == '4' or next_step == '0':
           R2_UserBased = gr(id_, top_n=5)
           next_step = input('If you would like to try new ingredients, press I. Press R for new recipes with the same ingredients or S to stop: ')
           if next_step == 'R' or next_step == 'r' or next_step == 'i' or next_step == 'I':
               continue
           elif next_step == 'S' or next_step == 's':
               print('Thanks for using the recipe finder')
               break
           else: 
               break
        else:
            continue
    elif next_step == 'R' or next_step == 'r':
        r1_table, r2_table = randomrecipeFinder()
        recipetables()
        next_step = input('Press 0, 1, 2, 3 or 4 to go into the recipe steps, Press R for new recipes  or Press I to change ingredients, or S to stop: ')
        try:
            next_step, id_ = nextSteps()                     
        except:
            pass
        if next_step == '1' or next_step == '2' or next_step == '3' or next_step == '4' or next_step == '0':
           R2_UserBased = gr(id_, top_n=5)
           next_step = input('If you would like to try new ingredients, press I. Press R for new recipes with the same ingredients or S to stop: ')
           if next_step == 'R' or next_step == 'r' or next_step == 'i' or next_step == 'I':
               continue
           elif next_step == 'S' or next_step == 's':
               print('Thanks for using the recipe finder')
               break
           else: 
               break
        else:
            continue
    elif next_step =='S' or 's':
        print('Thanks for using the recipe finder')
        break
    else:
        break
   
        
