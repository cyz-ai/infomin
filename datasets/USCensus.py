import os, time
import urllib
import os.path
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler
from collections import namedtuple

dirname = os.path.dirname(__file__)



def read_county(label='Proverty', sensitive_attribute='Gender'):
    #read in the data
    df = pd.read_csv('data/acs2015_county_data.csv')
    #drop the NAs
    df = df.dropna()
    df['Income'] = -np.log(df["Income"])
    if sensitive_attribute is 'Gender':
        # ratio of women 
        men_count = df['Men']
        women_count = df['Women']
        gender_ratio = men_count/(men_count + women_count)
        features = df.drop(['Poverty', 'State', 'CensusId', 'County', 'Men', 'Women'], axis=1)
        sensitive_attr = gender_ratio
    if sensitive_attribute is 'Color':
        # ratio of black people 
        features = df.drop(['Poverty', 'State', 'CensusId', 'County', 'Black', 'White', 'Asian'], axis=1).values
        n = len(features)
        sensitive_attr = df['Black'].values
        black = df['Black'].values
        white = df['White'].values
        asian = df['Asian'].values
        sensitive_attr = black/(black + white + asian + 1)
    n = len(features)
    features = np.atleast_2d(features).reshape(n, -1)
    labels = np.atleast_2d(df['Poverty'].values).reshape(-1, 1)
    sensitive_attr = np.atleast_2d(sensitive_attr).reshape(-1, 1)
    features = np.hstack((features, sensitive_attr))
    return features, labels, sensitive_attr


def read_tract(label='Proverty', sensitive_attribute='Gender'):
    #read in the data
    df = pd.read_csv('data/acs2015_census_tract_data.csv')
    #drop the NAs
    df = df.dropna()
    df['Income'] = -np.log(df["Income"])
    if sensitive_attribute is 'Gender':
        # ratio of women 
        men_count = df['Men']
        women_count = df['Women']
        gender_ratio = men_count/(men_count + women_count)
        features = df.drop(['Poverty', 'State', 'CensusTract', 'County', 'Men', 'Women'], axis=1)
        sensitive_attr = gender_ratio
    if sensitive_attribute is 'Color':
        # ratio of black people 
        features = df.drop(['Poverty', 'State', 'CensusTract', 'County', 'Black', 'White', 'Asian'], axis=1).values
        n = len(features)
        sensitive_attr = df['Black'].values
        black = df['Black'].values
        white = df['White'].values
        asian = df['Asian'].values
        sensitive_attr = black/(black + white + asian + 1)
    n = len(features)
    features = np.atleast_2d(features).reshape(n, -1)
    labels = np.atleast_2d(df['Poverty'].values).reshape(-1, 1)
    sensitive_attr = np.atleast_2d(sensitive_attr).reshape(-1, 1)
    features = np.hstack((features, sensitive_attr))
    return features, labels, sensitive_attr
    
    
  