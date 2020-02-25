import requests 
import json
import pandas as pd 
import numpy as np 

def get_trail_data(lat_lon_list, url, headers, radius=100, per_page=10):
    ''' Given a list of lat/lon tuples, send an api request for each lat/lon location,
    and return a pandas dataframe with all the response data.
    '''

    trails_df = pd.DataFrame()
    
    for i, (lat, lon) in enumerate(lat_lon_list):
        api_params = {'lat': lat,\
                    'lon': lon,\
                    'radius': radius,\
                    'per_page': per_page}

        response = get_api_response(url, headers, api_params)

        if response.status_code == 200:
            trails_df = append_response_to_df(trails_df, response, region_index=i)
        else:
            print(f'Bad response: {response.status_code}')     
       
    trails_df.drop_duplicates(subset='id', inplace=True)
    trails_df.reset_index(inplace=True, drop=True)
    return trails_df


def get_api_response(url, headers, api_params):
    response = requests.request("GET", url, headers=headers, params=api_params)
    return response


def append_response_to_df(df, response, region_index):
    r = response.json()['data']
    for i, trail in enumerate(r):
        df_row = pd.DataFrame(trail, index=[i])
        df_row['region_num'] = region_index
        df = df.append(df_row, ignore_index=True)
    return df


if __name__ == '__main__':

    url = 'https://trailapi-trailapi.p.rapidapi.com/trails/explore/'
    headers = {
        'x-rapidapi-host': "trailapi-trailapi.p.rapidapi.com",
        'x-rapidapi-key': "6734fb9f96msh2827b6ce0c2908dp1a382cjsn6b22445e5dba"
        }

    lat_lon_list = [(38.2544, -104.6091), # Pueblo
                    (39.7392, -104.9903), # Denver
                    (39.0639, -108.5506), # Grand Junction
                    (37.3489, -108.5859), # Cortez
                    (39.6403, -106.3742), # Vail
                    (37.6700, -106.6398), # South Fork
                    (40.4850, -106.8317)] # Steamboat

    st_trails_df = get_trail_data(lat_lon_list, url, headers, radius=100, per_page=1000)
    st_trails_df.to_pickle('../data/co_trails_df')