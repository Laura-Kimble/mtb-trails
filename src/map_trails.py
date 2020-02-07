import numpy as np
import pandas as pd
import datetime
import folium
from st_sklearn_nmf import get_topic_trails

def create_multiple_dots_layers(df, base_map, layer_names, trail_names_for_layers):
    ''' Create layers on a folium map.

    ARGS:
        df (dataframe): Dataframe with all data to be filtered for each layer, with 'lat' and 'lon' columns
        base_map (folium Map object): Map to apply the layers to
        layer_names (list of str): Names for the layers (topics)
        trail_names_for_layers (list of list of tuples): Each element in the list is a list of tuples with the 'top trails' for a topic.
            The first element in the tuple is the trail name, and the second is the region (not used in this mapping function).
    '''

    colors = ['red', 'orange', 'blue', 'purple', 'green', 'black']

    for idx, layer in enumerate(layer_names):
        trail_names = [trail for trail, region in trail_names_for_layers[idx]]
        layer_df = df[df['name'].map(lambda x: x in trail_names)]  
        color = colors[idx % 7]
        create_dots_layer(layer_df, base_map, layer, color=color)
        

def create_dots_layer(df, base_map, layer_name, color='black'):
    '''
    Create a dots layer onto the base map, with the lat/lon in the dataframe.
  
    ARGS:
        df (dataframe): Data to show in the layer, with 'lat' and 'lon' columns
        base_map (folium Map object): Base map to apply the layer to
        layer_name (str): Name of the layer.
        color (str): Color of the dots.
    '''

    feature_map = folium.FeatureGroup(name = layer_name)

    for i, row in df.iterrows():
        folium.CircleMarker(location=(row['lat'], row['lon']),
                                    radius=1.5,
                                    color=color,
                                    popup=str(row['name'] \
                                              + '\n\nDifficulty: '+ str(row['difficulty']) \
                                             ),
                                    fill=True).add_to(feature_map)

    base_map.add_child(feature_map)


if __name__ == '__main__':

    #Load the pickeled dataframes
    st_df = pd.read_pickle('../data/st_trails_df_2')
    W_df = pd.read_pickle('../models/W_df')

    top_topic_trails = get_topic_trails(W_df.values, n=50)
    topic_names = W_df.columns

    base_map = folium.Map(location=[39.73782,-98],
                            zoom_start=5,
                            tiles="Cartodbpositron")


    # Create map layers
    create_multiple_dots_layers(st_df, base_map, topic_names, top_topic_trails)

    # add toggle controls for the layers
    folium.LayerControl().add_to(base_map)

    base_map.save('../images/base_map.html')