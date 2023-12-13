# Adapted from https://frost.met.no/python_example.html
import requests
import datetime
import numpy as np

fp = "../Winddaten/Frost_API_credentials.txt"
with open(fp) as f:
    client_id = f.readline().rstrip('\n')
print(client_id)

start_time = datetime.datetime(2020,1,1)
end_time = datetime.datetime(2020,1,31)

filename_save = start_time.strftime("%Y%m%d") + "-" + end_time.strftime("%Y%m%d") + "_observations.csv"

end_time = end_time + datetime.timedelta(days=1)

time = start_time
while(time < end_time):
    print(time)
    next_day = time + datetime.timedelta(days=1)
    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    parameters = {
        'sources': 'SN76923',
        'elements': 'mean(wind_speed PT1M),wind_speed',
        'referencetime': time.strftime('%Y-%m-%d/')+next_day.strftime('%Y-%m-%d'),
    }
    # Issue an HTTP GET request
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    # Extract JSON data
    json = r.json()

    if r.status_code == 200:
        data = json['data']
        with open(filename_save, "a") as f:
            for x in data:
                np.savetxt(f, np.array([[x['referenceTime'], x['observations'][0]['value']]]), delimiter=';', fmt='%s')
    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])
        
    time += datetime.timedelta(days=1)