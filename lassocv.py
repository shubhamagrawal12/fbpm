#import requests

#payload = {"timestamp":"2015-09-17 20:50:00",
#           "description":"The Seattle Seahawks are a football team owned by Paul Allen."}
#r = requests.post(url="http://localhost:5000/model", data=payload)

#r.status_code

#r.json()

#***********************************************************

import sys
print("The following Python version was used to create this notebook:")
print(sys.version)

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
style.use('fivethirtyeight')

#%matplotlib inline


df = pd.read_json('data/posts.json', lines=True)
df['post_consumptions'] = df['insights'].apply(lambda x: x['post_consumptions']['values'][0]['value'])
df['post_consumptions_log'] = df['post_consumptions'].apply(lambda x: np.log(x+1))
df = df.set_index(pd.DatetimeIndex(df['created_time']))
df['description'] = df['description'].fillna('')

print(df.columns)

X = df[['created_time', 'description']].as_matrix()
print(X.shape)
print(X)

y_raw = df['post_consumptions_log'].as_matrix()
print(y_raw.shape)

y_scaler = StandardScaler()

y = y_scaler.fit_transform(y_raw.reshape(-1, 1))

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
	
pipe = make_pipeline(make_union(make_pipeline(FunctionTransformer(select_time_column, validate=False), dow_trans),
                              make_pipeline(FunctionTransformer(select_time_column, validate=False), month_trans),
                              make_pipeline(FunctionTransformer(select_text_column, validate=False), tfidf_vec)), 
							  LassoCV(n_alphas=200, cv=5, max_iter=2000, verbose=True, n_jobs=-1, random_state=None))
							 
pipe.fit(X, np.reshape(y, y_raw.shape))

joblib.dump(pipe, 'models/pipe.pkl')

df_mses = pipe.named_steps['lassocv'].mse_path_

pd.DataFrame([df_mses.mean(axis=1), np.log(pipe.named_steps['lassocv'].alphas_)], index=['mse','log_alpha']).transpose().plot(x='log_alpha', y='mse')
plt.show()

p2 = joblib.load('models/pipe.pkl')

#pd.DataFrame(p2.predict(X)).hist(bins=100)
#plt.show()

X_test = np.array([[pd.Timestamp('2015-09-17 20:50:00'),
                    'The Seattle Seahawks are a football team owned by Paul Allen.']])
					
print(X_test.shape)

print(p2.predict(X_test))