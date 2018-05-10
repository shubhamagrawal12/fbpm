#from sklearn.externals import joblib

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LassoCV
from sklearn.externals import joblib

import matplotlib.pyplot as plt
from matplotlib import style


class DayOfWeekTransformer(TransformerMixin):
    
    def __init__(self):
        self.one_hot = OneHotEncoder()
    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns=['ct'])
        df_dow = df['ct'].apply(lambda x: x.dayofweek)
        self.one_hot.fit(df_dow.values.reshape(-1, 1))
        return self
    
    def transform(self, X, **transform_params):
        df = pd.DataFrame(X, columns=['ct'])
        df_dow = df['ct'].apply(lambda x: x.dayofweek)
        return self.one_hot.transform(df_dow.values.reshape(-1, 1))
		
class MonthTransformer(TransformerMixin):
    
    def __init__(self):
        self.one_hot = OneHotEncoder()

    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns=['ct'])
        df_dow = df['ct'].apply(lambda x: x.month)
        self.one_hot.fit(df_dow.values.reshape(-1, 1))
        return self
    
    def transform(self, X, **transform_params):
        df = pd.DataFrame(X, columns=['ct'])
        df_dow = df['ct'].apply(lambda x: x.month)
        return self.one_hot.transform(df_dow.values.reshape(-1, 1))
		
dow_trans = DayOfWeekTransformer()
month_trans = MonthTransformer()
tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)


def select_time_column(X):
    return X[:,0]


def select_text_column(X):
    return X[:,1]


p2 = joblib.load('models/pipe.pkl')
#p2 = joblib.load('models/ridgedpipe.pkl')

X_test = np.array([[pd.Timestamp('2015-09-17 20:50:00'),
                    'The Seattle Seahawks are a football team owned by Paul Allen.']])
					
print(X_test.shape)

print(p2.predict(X_test))
