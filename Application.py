# import the necessary packages

from flask import Flask, abort, request
import numpy as np
import pandas as pd
from functools import wraps
from sklearn.externals import joblib
from sklearn.base import TransformerMixin
from flask import jsonify, make_response
import json
import werkzeug
from werkzeug.exceptions import HTTPException, NotFound



app = Flask(__name__)

# Class definition for converting date/month into vectors
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
		
		
def select_time_column(X):
		return X[:,0]

def select_text_column(X):
	return X[:,1]
	
def successResponse(message):
	return jsonify({
	"STATUS" : "SUCCESS", 
	"result" : message
	})

def errorResponse(message):
	return jsonify({
	"STATUS" : "ERROR", 
	"message" : message
	})
	
# Controller using Decorated function	
def vlidateLassoCVApi(f):
	@wraps(f)
	def decorated_function(*args, **kwargs):
		print("Hit decoreator")
		a = (('timestamp') in request.form)
		b = (('description') in request.form)
		if(a and b ):
			print("continue")
		else:
			print("bad request no data")
			abort(make_response(errorResponse("Seems like Timestamp or Description is not there"), 400))
		print(request.form['timestamp'], request.form['description'])
		return f(*args, **kwargs)
	return decorated_function
		
# To run the given model (LassoCV Algorithm)		
@app.route('/api/v1/lassocv', methods=['POST'])
@vlidateLassoCVApi
def testData():

	print("Reached1 new")
	postDataTime = request.form['timestamp']
	postDataDesc =  request.form['description']
	p2 = joblib.load('../models/pipe.pkl')
	X_test = np.array([[pd.Timestamp(postDataTime), postDataDesc]])
	pridictedValue = p2.predict(X_test)
	result = "".join(str(x) for x in pridictedValue)
	return successResponse(result)

### To run the model created using ridgeCV algorithm	
@app.route('/api/v1/ridgecv', methods=['POST'])
@vlidateLassoCVApi
def ridgeData():

	postDataTime = request.form['timestamp']
	postDataDesc =  request.form['description']
	p2 = joblib.load('../models/ridgedpipe.pkl')
	X_test = np.array([[pd.Timestamp(postDataTime), postDataDesc]])
	pridictedValue = p2.predict(X_test)
	result = "".join(str(x) for x in pridictedValue)
	return successResponse(result)

##### Application level Error Handling #####
@app.errorhandler(werkzeug.exceptions.HTTPException)
def handle_bad_request(e):
    return errorResponse('bad request!'),400
	
@app.errorhandler(werkzeug.exceptions.NotFound)
def handle_bad_request(e):
    return errorResponse('Not found request!'),404
	
## Initiating the Server	
if __name__=='__main__':
	#app.run(debug=True)
	print("Server Has Started !!")
	app.run(host= '0.0.0.0', port=5000, debug=True)

