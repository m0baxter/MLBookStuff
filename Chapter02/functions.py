
import os
import tarfile
from six.moves import urllib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def getHousingData( url = HOUSING_URL, path = HOUSING_PATH ):
    """Retrieves the housing data from url and saves it to path."""

    if ( not os.path.isdir(path) ):
        os.makedirs(path)

    tgzPath = os.path.join( path, "housing.tgz" )
    urllib.request.urlretrieve( url, tgzPath )
    housing_tgz = tarfile.open( tgzPath )
    housing_tgz.extractall( path = path )
    housing_tgz.close()

def loadData( path= HOUSING_PATH ):
    """Loads the data into a pandas dataframe."""

    csvPath = os.path.join( path, "housing.csv")

    return pd.read_csv(csvPath)


def argmaxTopN( arr, N ):
    """Returns the arguments of the largest N elements of N unsorted"""

    return np.argpartition(arr, -N)[-N:]


# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

class SelectBestFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, importances, k ):
        self.importances = importances
        self.num = k

    def fit(self, X, y = None):
        self.feature_indices_ = argmaxTopN( self.importances, self.k )
        return self

    def transform(self, X):
        return X[:, self.feature_indices_]

