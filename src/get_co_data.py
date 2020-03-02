import requests 
import pandas as pd 
import numpy as np 
from math import sin, cos, sqrt, atan2, radians


def get_trail_data(lat_lon_list, url, headers, radius=100, per_page=10):
    ''' Given a list of lat/lon tuples, send an api request to singletracks.com
     for each lat/lon location, and return a pandas dataframe with all the response data.
    '''

    trails_df = pd.DataFrame()
    
    for i, (lat, lon) in enumerate(lat_lon_list):
        api_params = {'lat': lat,
                      'lon': lon,
                      'radius': radius,
                      'per_page': per_page}

        response = get_api_response(url, headers, api_params)

        if response.status_code == 200:
            trails_df = append_response_to_df(trails_df, response, region_index=i)
        else:
            print(f'Bad response: {response.status_code}')     
       
    trails_df.drop_duplicates(subset='id', inplace=True)
    trails_df.reset_index(inplace=True, drop=True)
    trails_df = create_derived_cols(trails_df)
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


def create_derived_cols(df):
    df['length_rounded'] = df['length'].apply(lambda x: np.round(float(x), 0))
    df['rating_rounded'] = df['rating'].apply(lambda x: np.round(float(x), 1))
    df['description_length'] = df['description'].apply(lambda x: len(x))

    diff_dict = {'novice': 1, 'beginner': 2, 'intermediate': 3, 'advanced': 4, 'expert': 5, None: 3}
    df['difficulty_num'] = df['difficulty'].map(lambda x: diff_dict[x])

    region_dict = {0: "Pueblo", 1: 'Denver', 2: 'Grand Junction', 3: 'Cortez', 4: "Vail", 5: "South Fork", 6: 'Steamboat'}
    df['region_name'] = df['region_num'].map(lambda x: region_dict[x])
    df['dist_to_Denver_km'] = df.apply(lambda row: calc_dist_to_Denver_km(np.float(row['lat']), np.float(row['lon'])), axis=1)

    df['Pump_track'] = df['features'].map(lambda x: 'Pump' in x)
    df['Lift_service'] = df['features'].map(lambda x: 'Lift' in x)
    return df


def calc_dist_to_Denver_km(lat, lon):

    # approximate radius of earth in km
    R = 6373.0
    lat1 = radians(39.7392)  # Denver
    lon1 = radians(-104.9903)
    lat2 = radians(lat) 
    lon2 = radians(lon)
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


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