import numpy as np
import pandas as pd
import datetime
import folium
from co_sklearn_nmf import get_co_topic_trails
from map_trails import create_multiple_dots_layers


if __name__ == '__main__':

    # Load the pickeled dataframes for CO trails
    st_df = pd.read_pickle('../data/co_trails_df')
    W_df = pd.read_pickle('../models/co_W_df')

    top_topic_trails = get_co_topic_trails(W_df.values, n=50)
    topic_names = W_df.columns

    co_map = folium.Map(location=[39, -105],
                        zoom_start=7,
                        tiles="Cartodbpositron")

    # Create map layers
    create_multiple_dots_layers(st_df, co_map, topic_names, top_topic_trails)

    # add toggle controls for the layers
    folium.LayerControl().add_to(co_map)

    co_map.save('../images/co_map.html')
