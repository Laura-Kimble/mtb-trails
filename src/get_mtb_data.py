
import requests 
import json
import pandas as pd 
import numpy as np 


def get_trail_data(lat_lon_list, url, request_type, api_key, maxDistance=30, maxResults=10):
    ''' Given a list of lat/lon tuples, send an api request for each lat/long location,
    and return a pandas dataframe with all the response data.
    '''

    trails_df = pd.DataFrame()
    
    for lat, lon in lat_lon_list:
        api_params = {'key': api_key,\
                'lat': lat,\
                'lon': lon,\
                'maxDistance': maxDistance,\
                'maxResults': maxResults}

        response = get_api_response(url, request_type, api_params)

        if response.status_code == 200:
            trails_df = append_response_to_df(trails_df, response)
        else:
            print(f'Bad response: {response.status_code}')        

    return trails_df


def get_api_response(url, request_type, api_params):
    response = requests.get(f'{url}/{request_type}', api_params)
    return response


def append_response_to_df(df, response):
    r = response.json()
    for i, trail in enumerate(r['trails']):
        df_row = pd.DataFrame(trail, index=[i])
        df = df.append(df_row, ignore_index=True)
    return df


if __name__ == '__main__':

    api_key = '7022841-4e9480acc62afb16e4bd72d1558707b7'
    url = 'https://www.mtbproject.com/data'
    request_type = 'get-trails'
    maxDistance = 100
    maxResults = 500

    lat_lon_list = [(45.5051, -122.6750), # Portland
                    (39.7392, -104.9903), # Denver
                    (38.5733, -109.5498), # Moab
                    (40.4406, -79.9959), # Pittsburgh
                    (34.0522, -118.2437), # Los Angeles
                    (48.7519, -122.4787), # Bellingham WA
                    (43.6150, -116.2023), # Boise
                    (33.7490, -84.3880), # Atlanta
                    (34.7465, -92.2896), # Little Rock
                    (42.9634, -85.6681)] # Grand Rapids

    trails_df = get_trail_data(lat_lon_list, url, request_type, api_key, maxDistance=maxDistance, maxResults=maxResults)
    trails_df.to_pickle('../data/mtb_trails_df')



    


