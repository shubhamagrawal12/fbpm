import requests

# Defining the Payload
payload = {"timestamp":"2015-09-17 20:50:00",
          "description":"The Seattle Seahawks are a football team owned by Paul Allen."}
		   
# Original model based on LassoCV algorithm - Version 1.0
r = requests.post(url="http://localhost:5000/api/v1/lassocv", data=payload)

# Second model using RidgeCV algorithm - Version 1.0
#r = requests.post(url="http://localhost:5000/api/v1/ridgecv", data=payload)

# Returning the stats code 
print(r.status_code)

# Printing the result
print(r.json())