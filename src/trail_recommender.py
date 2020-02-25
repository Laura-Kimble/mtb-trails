import numpy as np
import pandas as pd 
import item_recommender as ir
from sklearn.metrics.pairwise import cosine_similarity
from co_sklearn_nmf import get_co_trail_names, get_co_index
from random import choice, randint, sample
import importlib
importlib.reload(ir)

# Make X matrix from description topic features (W matrix) plus other features (length, difficulty, etc.)
# Standardize X matrix
# Assign weightings to features for recommender
# Get and filter recommendations

def filter_reccos(recco_items, recco_ids, filter_criteria, df):
    ''' Filter the recommendations by a set of criteria, using data fields in the original df.
    INPUTS:
        recco_items: List of names of the recommendations
        recco_ids: List of id's for the recommendations, to match to the original df
        filter_criteria: Dictionary where key is the column name in the df to filter on, and values is a list of accepted values in that col
        df: Dataframe with the additional data columns for filtering, including an id column to lookup the reccos
    OUTPUTS:
        filtered_items: List of filtered item names
        filtered_ids: List of filtered ids
    '''
    filtered_df = df[df['id'].map(lambda x: x in recco_ids)]
    for col_name, values in filter_criteria.items():
        filtered_df = filtered_df[filtered_df[col_name].map(lambda x: x in values)]
    filtered_ids = np.array(filtered_df['id'])
    filtered_items = np.array(recco_items)[np.isin(np.array(recco_ids), filtered_ids)]
    return list(filtered_items), list(filtered_ids)


if __name__ == '__main__':
    W_df = pd.read_pickle('../models/co_W_df')
    st_df = pd.read_pickle('../data/co_trails_df_2')
    st_df_with_desc = st_df[st_df['description_length']>=40]
    feature_cols = ['length_rounded', 'rating_rounded', 'difficulty_num']  # list out feature cols to use in model --
    # need to do some featurizing (one-hot encode Lift, pump... difficulty num)
    feature_cols.extend(list(W_df.columns))

    X = pd.merge(st_df_with_desc, W_df, on='id')[feature_cols]
    trail_names = list(get_co_trail_names())
    trail_ids = list(get_co_index())

    trail_recommender = ir.ItemRecommender(similarity_measure=cosine_similarity)
    trail_recommender.fit(X, trail_names=trail_names, trail_ids=trail_ids)

    rand_trail = choice(trail_names)
    recco_trails, recco_trail_ids = trail_recommender.get_recommendations(rand_trail, n=15)

    filter_criteria_dict = {'difficulty': ['intermediate', 'advanced']}
    filtered_names, filtered_ids = filter_reccos(recco_trails, recco_trail_ids, filter_criteria_dict, st_df_with_desc)
    print(f'Your trail is {rand_trail}:\n\n')
    print(f'Similar trails are:\n{recco_trails}\n\n')
    print(f'Filtered set of recommendations:\n{filtered_names}')

    rand_n = randint(2, 10)
    rand_user_trails = sample(set(trail_names), rand_n)
    user_recs = trail_recommender.get_user_recommendation(rand_user_trails, n=5)
    print(f'User trails are:\n{rand_user_trails}\n\n')
    print(f'Recommendations for user are:\n{user_recs}')