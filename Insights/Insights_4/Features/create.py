import numpy as np
import pandas as pd
import ast
from collections import Counter
import json
from pandas.io.json import json_normalize
import unicodedata
import re

def addDateFeatures(data):
    """ Adds extra date features like: day, month, quarter and year """
    data['release_year'] = pd.to_datetime(data['release_date']).dt.year
    data['release_quarter'] = pd.to_datetime(data['release_date']).dt.quarter
    data['release_month'] = pd.to_datetime(data['release_date']).dt.month
    data['release_day'] = pd.to_datetime(data['release_date']).dt.day

    data['release_year'].loc[data['release_year'] > 2019] -= 100 # Fixing release years.

    # Filling the columns NaN values (only one NaN value exist in test data) with mode of the data
    data['release_year'].fillna(data['release_year'].mode()[0],inplace = True)
    data['release_quarter'].fillna(data['release_quarter'].mode()[0],inplace = True)
    data['release_month'].fillna(data['release_month'].mode()[0],inplace = True)
    data['release_day'].fillna(data['release_day'].mode()[0],inplace = True)

    data['release_year'] = data['release_year'].astype(int)
    data['release_quarter'] = data['release_quarter'].astype(int)
    data['release_month'] = data['release_month'].astype(int)
    data['release_day']= data['release_day'].astype(int)

    # Reshape the release date column to from this format: 'month-day-year' to this format: 'year-month-day'
    data['release_date'] = data['release_month'].astype(str) + '/' + data['release_day'].astype(str) + '/' + data['release_year'].astype(str)
    data['release_date'] = pd.to_datetime(data['release_date'], format="%m/%d/%Y")
    data['release_day_of_week'] = data['release_date'].dt.dayofweek # The day of the week with Monday = 0, Sunday = 6


def addCollectionFeatures(data):
    """ Adds belongs to collection features """
    data['belongs_to_collection'] = stringToJSON(data['belongs_to_collection'])
    data['collection_name'] = data['belongs_to_collection'].apply(lambda entry: entry['name'] if entry != {} else 0)
    data['collection_count'] = data['belongs_to_collection'].apply(lambda entry: len(entry) if entry != {} else 0)

def addExtraFeatures(data):
    """ Adds extra features """
    pass

def stringToJSON(data):
    """ change strings to JSON format for a pandas coloumn """
    data.fillna(value = '[{}]',inplace = True) # JSON empty string format
    data.loc[data == '[]'] = '[{}]'
    return data.apply(lambda entry: re.compile(r'\\x([0-9a-fA-F]{2})').sub('',entry.replace("None",'"ddd"'))).apply(ast.literal_eval)

def addGenreFeatures(data):
    """ Adds extra features """
    data['genre'] = stringToJSON(data['genres'])
    train_genres = pd.DataFrame(
    {
        'id' : data['id'].values.repeat(data['genres'].str.len(), axis = 0),
        'genre' : np.concatenate(data['genres'].tolist())
    })

    train_genres['genre'] = train_genres['genre'].map(lambda genre: genre.get('name'))
    train_genres = train_genres.set_index('id').genre.str.get_dummies().sum(level = 0)
    data = pd.merge(data, train_genres, on = 'id', how = 'outer')

def is_en(df):
    df['is_english'] = 0
    df.loc[train['original_language']=="en", 'is_english'] = 1
    return df

def cast_preproccessing(train):
    cast_list = []
    male = 0
    female = 0
    Hero_male = 0
    Hero_female = 0

    for x in train['cast']:
        try:
            for item in ast.literal_eval(x):
                cast_list.append(item['name'])

                if item['order'] == 0:
                    if item['gender'] == 1:
                        Hero_female += 1
                    else:
                        Hero_male += 1

                if item['gender'] == 1:
                    female += 1
                else:
                    male += 1
        except:
            continue
    return cast_list,male,female,Hero_male,Hero_female

def is_famous(df):
    output = cast_preproccessing(df)
    c = Counter(output[0])
    list_freq = [i[0] for i in c.most_common(50)]
    for i in list_freq[:9]:
        df[i] = 0
    df['is Famous'] = 0
    for idx, row in df.iterrows():
        try:
            for item in ast.literal_eval(row['cast']):
                if item['name'] in list_freq:
                    df['is Famous'][idx]=1
                if item['name'] in list_freq[:9]:
                    df[item['name']][idx]=1
        except:
            continue
    return df

def add_production_companies_features(df, top_production_companies):
    df['num_companies'] = df['production_companies'].apply(lambda x: len(x) if x != {} else 0)
    df['all_companies'] = df['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
    for company in top_production_companies:
        df[ 'is_by_'+company] = df['all_companies'].apply(lambda x: 1 if company in x else 0)
    
    return df