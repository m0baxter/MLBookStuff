
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def findTitles( data ):
    """Returns a set of all honourifics present in the data set"""

    titles = set([])

    for name in data["Name"]:
        for word in name.split():
            if ("." in word):
                titles.add(word)

    return titles

def titleStats( data, title, stat = "mean" ):
    """Determines the mean age of all people with the given title."""

    if ( not (stat in ["mean", "median", "mode"]) ):
        print("Not a statistical measure.")

        return

    ages = []

    for i in range(len(data)):

        if ( title in data["Name"][i] ):
            ages.append( data["Age"][i] )

    if ( stat == "mean" ):
        return pd.Series(ages).mean()

    elif ( stat == "median" ):
        return pd.Series(ages).median()

    else:
        return pd.Series(ages).mode()

def titleMeans( data ):
    """Generates a ditionary of the average age for a given honourific."""

    titles = findTitles( data )

    means = {}

    for t in titles:
        means[t] = titleStats( data, t, stat = "mean" )

    return means

def inferAge( data, titleDict ):
    """Fills in missing age data using the average age for a given honourific."""

    for i in range(len(data)):

        if ( pd.isnull(data["Age"][i]) ):
            for title in titleDict.keys():
                if ( title in data["Name"][i] ):
                    data.at[i, "Age"] = titleDict[title]

    return

def inferFares( data, fareDict ):
    """Fills in missing fare data using the avrage fare for each ticket class."""

    for i in range(len(data)):

        if ( pd.isnull(data["Fare"][i]) ):
            data.at[i, "Fare"] = fareDict[ data["Pclass"][i] ]

    return

def findFamilies( data, minSize = 3 ):
    """Finds family units based on the data that contain minSize or more members (There may
       be more members not in this portion of the data, this shuould be kept in mind)."""
    
    families = {}
    
    for i in range( len(data) ):
        if ( data["FamSize"][i] > 0):
            w = data["Name"][i].split(",")
            
            if ( w[0] in families ):
                families[ w[0] ].append(i)

            else:
                families[ w[0] ] = [i]

    toRemove = []

    for f in families:
        if ( len(families[f]) < minSize ):
            toRemove.append(f)

    for k in toRemove:
        families.pop(k, None)

    return families

def addFamily( data, families ):
    """Adds a column of families to the data set. Entries not found to be members of a
       family are marked as ***."""

    data["Family"] = pd.Series( dtype = str )

    for f in families:
        for index in families[f]:

            data.at[index, "Family"] = f

    data.at[ pd.isnull(data["Family"]), "Family" ] = "***"

    return

def addFamilyTest( data, families):
    """Adds family column to the test data."""

    data["Family"] = pd.Series( dtype = str )

    for i in range( len(data) ):

        w= data["Name"][i].split(",")

        if ( w[0] in families and ( data["FamSize"][i] > 0 ) ):
            data.at[i, "Family"] = w[0]

    data.at[ pd.isnull(data["Family"]), "Family" ] = "***"

    return

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

