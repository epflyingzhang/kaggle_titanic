# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:51:41 2020

@author: Ying
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# load raw data
data_dir = 'data/titanic/'
print("current working directory:", os.getcwd())
print(os.listdir(data_dir))  # 

train_file = data_dir + 'train.csv'
test_file = data_dir + 'test.csv'

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)


# missing value in column Age (177/891), Cabin (687/891) and Embarked (2/891), Fare(1 missing in Test)
def clean_df(df):
    df['Age'] = df['Age'].fillna(df_train['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df_train['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df_train['Fare'].median())
    return df
    
df_train = clean_df(df_train)
df_test = clean_df(df_test)
df_train.info()

# clean cabin data
import re
def lambda_cabin_section(x):
     return 1 if pd.notnull(x) else 0

def lambda_cabin_number(x):
    # find the number after the first letter, before any space
    num_str = re.findall('^[A-Z]([0-9]*)', x.split(" ")[0])[0] if pd.notnull(x) else ''
    return (int(num_str) if num_str!='' else -1)
df_train['cabin_section'] = df_train['Cabin'].apply(lambda_cabin_section)
df_train['cabin_number'] = df_train['Cabin'].apply(lambda_cabin_number)

df_test['cabin_section'] = df_test['Cabin'].apply(lambda_cabin_section)
df_test['cabin_number'] = df_test['Cabin'].apply(lambda_cabin_number)

sns.countplot(x='cabin_section', data=df_train)
plt.show()
sns.barplot(x='cabin_section', y='Survived', data=df_train)
plt.show()
# the ones without cabin has significantly low survival rate

sns.boxplot(x='Survived', y='cabin_number', data=df_train[df_train['cabin_section']==1])
plt.show()
# cabin_number has no predictive power

# binning Age
def lambda_age_bin(x):
    if x < 5:
        return 'kid (0-6)'
    elif x < 18:
        return 'teenager (6-18)'
    else:
        return 'adult (>15)'
df_train['Age_group'] = df_train['Age'].apply(lambda_age_bin)
df_test['Age_group'] = df_test['Age'].apply(lambda_age_bin)

df_train.Age.hist()
plt.show()
sns.countplot(x='Age_group', data=df_train)
plt.show()
sns.barplot(x='Age_group', y='Survived', data=df_train)
plt.show()



# log transform Fare
df_train['Fare_log'] = df_train['Fare'].apply(np.log1p)
df_test['Fare_log'] = df_test['Fare'].apply(np.log1p)
sns.boxplot(x='Survived', y='Fare_log', data=df_train)
plt.show()

# zero fare
df_train['Fare_zero'] = df_train['Fare']==0
df_test['Fare_zero'] = df_train['Fare']==0
sns.countplot(x='Fare_zero', data=df_train)
plt.show()
sns.barplot(x='Fare_zero', y='Survived', data=df_train)
plt.show()

# name length
df_train['name_len'] = df_train['Name'].apply(len)
df_test['name_len'] = df_test['Name'].apply(len)
sns.boxplot(x='Survived', y='name_len', data=df_train)
plt.show()

# name title
special_name_titles = ['Master.', 'Don.', 'Rev.', 'Dr.', 'Mme.',
                       'Major.', 'Lady.', 'Sir.', 'Mlle.', 'Col.', 'Capt.']
df_train['name_title'] = df_train['Name'].apply(lambda x: re.findall( ',\s([a-zA-Z]+\.)', x)).apply(lambda x: x[0] if len(x) > 0 else 'No_titile')
df_test['name_title'] = df_test['Name'].apply(lambda x: re.findall( ',\s([a-zA-Z]+\.)', x)).apply(lambda x: x[0] if len(x) > 0 else 'No_titile')

for title in special_name_titles:
    df_train['name_title_'+title] = df_train['name_title']==title
    df_test['name_title_'+title] = df_train['name_title']==title

# combine SibSp and Parch to family
df_train["family"] = df_train["SibSp"] + df_train["Parch"] + 1
df_test["family"] = df_test["SibSp"] + df_test["Parch"] + 1
plt.show()
sns.countplot(x='family', data=df_train)
plt.show()
sns.barplot(x='family', y='Survived', data=df_train)
plt.show()


# other ideas
    # 
#def title_corr(t):
#    newt = t
#    if t == 'Mrs' or t == 'Mr' or t == 'Miss':
#        return newt
#    elif t == 'Capt' or t == 'Col' or t == 'Major' or t == 'Dr' or t == 'Rev':
#        newt = 'Crew'
#    elif t == 'Jonkheer' or t == 'Sir' or t == 'the Countess' or t == 'Lady' or t == 'Master':
#        newt = 'Noble'
#    elif t == 'Don':
#        newt = 'Mr'
#    elif t == 'Dona' or t == 'Ms' or t == 'Mme':
#        newt = 'Mrs'
#    elif t == 'Mlle':
#        newt = 'Miss'
#    else: print("Title not included:", t)
#    return newt
    
# use a different way to fill missing Age
#def calc_age(df, cl, sx, tl):
#    a = df.groupby(["Pclass", "Sex", "Titles"])["Age"].median()
#    return a[cl][sx][tl]
    



# save cleaned data
df_train.to_csv('data/df_train_clean.csv', index=False)
df_test.to_csv('data/df_test_clean.csv', index=False)
