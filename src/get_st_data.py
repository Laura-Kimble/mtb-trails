import requests 
import pandas as pd 
import numpy as np 


def get_trail_data(lat_lon_list, url, headers, radius=100, per_page=10):
    ''' Given a list of lat/lon tuples, send an api request to singletracks.com
     for each lat/lon location, and return a pandas dataframe with all the response data.
    '''

    trails_df = pd.DataFrame()
    
    for lat, lon in lat_lon_list:
        api_params = {'lat': lat,
                      'lon': lon,
                      'radius': radius,
                      'per_page': per_page}

        response = get_api_response(url, headers, api_params)

        if response.status_code == 200:
            trails_df = append_response_to_df(trails_df, response)
        else:
            print(f'Bad response: {response.status_code}')        

    return trails_df


def get_api_response(url, headers, api_params):
    response = requests.request("GET", url, headers=headers, params=api_params)
    return response


def append_response_to_df(df, response):
    r = response.json()['data']
    for i, trail in enumerate(r):
        df_row = pd.DataFrame(trail, index=[i])
        df = df.append(df_row, ignore_index=True)
    return df


if __name__ == '__main__':

    url = 'https://trailapi-trailapi.p.rapidapi.com/trails/explore/'
    headers = {
        'x-rapidapi-host': "trailapi-trailapi.p.rapidapi.com",
        'x-rapidapi-key': "6734fb9f96msh2827b6ce0c2908dp1a382cjsn6b22445e5dba"
        }

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

    st_trails_df = get_trail_data(lat_lon_list, url, headers, radius=100, per_page=100)
    st_trails_df.to_pickle('../data/st_trails_df')