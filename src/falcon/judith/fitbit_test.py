import os, sys
import cherrypy
cherrypy.config.update({'server.socket_port': 8081})
fitbit_api_path = os.environ.get('fitbitapi_dir')
sys.path.append(fitbit_api_path)
import fitbit
import gather_keys_oauth2 as Oauth2
import pandas as pd 
import datetime



fitbit_api_path = os.environ.get('fitbitapi_dir')
sys.path.append(fitbit_api_path)

CLIENT_ID = os.environ.get('FITBIT_OAUTH_CLIENTID')
CLIENT_SECRET = str(os.environ.get('FITBIT_OAUTH_CLIENTSECRET'))

assert CLIENT_ID is not None

server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
#print(server)
#server.browser_authorize()
#sys.exit()

#ACCESS_TOKEN = str(server.fitbit.client.session.token['access_token'])
#REFRESH_TOKEN = str(server.fitbit.client.session.token['refresh_token'])
ACCESS_TOKEN='eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMkJNV04iLCJzdWIiOiI3MzJSUzMiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJhY3QgcnNldCBybG9jIHJ3ZWkgcmhyIHJudXQgcnBybyByc2xlIiwiZXhwIjoxNTkyODc4NDY5LCJpYXQiOjE1OTI4NDk2Njl9.DIo9E7VJfi5gsCUs80to4uGPViZrSRZeePXnY0A7zg8'
REFRESH_TOKEN='2604bdafe12e7928065b7c71f738a4d6c0cbda54622424dff10c4c931b74f780'
auth2_client = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET, oauth2=True, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)

yesterday = str((datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
yesterday2 = str((datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d"))
yesterday3 = str((datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"))
today = str(datetime.datetime.now().strftime("%Y-%m-%d"))

heartrates = []
days = [yesterday3, yesterday2, yesterday, today]
print(days)
for day in days:
    #fitbit_stats = auth2_client._COLLECTION_RESOURCE('activities/heart', date=day)
    #print(fitbit_stats) 
    fit_statsHR = auth2_client.intraday_time_series('activities/heart', base_date=day, detail_level='1sec')
    heart_stuff = fit_statsHR['activities-heart'][0]
    resting_heart_rate = heart_stuff['value']['restingHeartRate']
    heartrates.append(resting_heart_rate)
    print('Appended')

print(heartrates)    



#print(fit_statsHR['activites-heart-intraday'])

'''
{'value': 
   {'heartRateZones': [{'caloriesOut': 2430.22428, 'name': 'Out of Range', 'max': 95, 'minutes': 1356, 'min': 30}, 
                       {'caloriesOut': 409.02224, 'name': 'Fat Burn', 'max': 133, 'minutes': 62, 'min': 95}, 
                       {'caloriesOut': 0, 'name': 'Cardio', 'max': 162, 'minutes': 0, 'min': 133}, 
                       {'caloriesOut': 0, 'name': 'Peak', 'max': 220, 'minutes': 0, 'min': 162}], 
                       'customHeartRateZones': [], 
                       'restingHeartRate': 54}, 

                       'dateTime': '2020-06-21'}
'''
