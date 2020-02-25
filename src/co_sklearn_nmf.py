import numpy as np 
import pandas as pd 
import joblib
from sklearn.decomposition import NMF
from gensim.parsing.preprocessing import STOPWORDS
from mtb_sklearn_lda import make_vectorizer, transform_vectorizer, get_top_words
from st_sklearn_nmf import print_top_topic_words, print_topic_trails, fit_nmf
import matplotlib.pyplot as plt

def get_co_descriptions():
    st_df = pd.read_pickle('../data/co_trails_df')
    st_df_with_desc = st_df[st_df['description_length']>=40]
    return st_df_with_desc['description']

def get_co_trail_names():
    st_df = pd.read_pickle('../data/co_trails_df')
    st_df_with_desc = st_df[st_df['description_length']>=40]
    return st_df_with_desc['name']

def get_co_regions():
    st_df = pd.read_pickle('../data/co_trails_df')
    st_df_with_desc = st_df[st_df['description_length']>=40]
    return st_df_with_desc['region_name']

def get_co_index():
    st_df = pd.read_pickle('../data/co_trails_df')
    st_df_with_desc = st_df[st_df['description_length']>=40]
    return st_df_with_desc['id']


def get_co_topic_trails(W, n=10):
    all_topic_trails = []
    trail_names = get_co_trail_names()
    regions = get_co_regions()
    
    for topic in W.T:
        top_ind = topic.argsort()[:-n-1:-1]
        top_trails = trail_names[top_ind]
        top_regions = regions[top_ind] 
        all_topic_trails.append(list(zip(top_trails, top_regions)))
    return all_topic_trails


if __name__ == '__main__':
    
    # Get singletracks trail summary data for Colorado trails
    X = get_co_descriptions()

    # Create  set of stopwords to use 
    my_stopwords = set(['climb', 'mountain', 'road', 'singletrack', 'loop', 'trail', 'trails',  'ride', 'area', 'route', 'way', \
                        'feature', 'section','sections', 'riding', 'll', 'rd', 'going', 'stay', \
                    'loop','br', 'mile', 'miles', 'right', 'left', 'www', 'http', 'https', 'bike', 'bikes', 'bicycle', 'bicycles', \
                    'continue', 'rider', 'riders', 'parking', 'lot', 'turn', 'start', 'starts', 'description', 'cross', \
                    'north', 'south', 'east', 'west', '-PRON-', 'pron', 'nee', 'regard', 'shall', 'use', 'win', \
                    'park', 'point', 'biking', 'follow', 'single', 'track', 'intersection', 'trailhead', 'head', \
                    'good', 'great', 'nice', 'time', 'include', 'place', 'come', 'downhill', 'look', 'near'])
    bitri_stops = set(['parking_lot', 'trail_starts', 'mile_turn', 'north_south', 'mountain_bike', 'mountain_biking', 'single_track', \
                    'mountain_bike_trail', 'trail_head'])
    all_stopwords = my_stopwords.union(STOPWORDS).union(bitri_stops)

    # Create TF-IDF matrix
    tfidf_vect = make_vectorizer(X, tf_idf=True, lemmatize=False, max_df=0.8, min_df=2, max_features=1000, stop_words=all_stopwords,\
                                ngram_range=(1, 2))
    tfidf = transform_vectorizer(tfidf_vect, X)
    # top_words = get_top_words(tfidf_vect, tfidf, n=50)
    
    feature_names = np.array(tfidf_vect.get_feature_names())
    trail_names = get_co_trail_names()
    trail_ids = get_co_index()

    # NMF
    k = 10
    topics = ['topic_{}'.format(i) for i in range(k)]

    nmf, W, H = fit_nmf(tfidf, n_topics=k, alpha=0.)
    W_df = pd.DataFrame(W, index=trail_ids, columns=topics)
    H_df = pd.DataFrame(H, index=topics, columns=feature_names)
    print(f'reconstruction error: {nmf.reconstruction_err_:0.3}')

    print_top_topic_words(H, feature_names, n=10)
    print('\n\n')
    all_topic_trails = get_co_topic_trails(W, n=10)
    print_topic_trails(all_topic_trails)

    # Dump vectorizer, model and W and H as dataframes
    joblib.dump(nmf, '../models/co_nmf_model.joblib')
    joblib.dump(tfidf_vect, '../models/co_tfidf_vec.joblib')

    # W_df = pd.DataFrame(W, index=trail_names, columns=topics)
    # H_df = pd.DataFrame(H, index=topics, columns=feature_names)
    W_df.to_pickle('../models/co_W_df')
    H_df.to_pickle('../models/co_H_df')

    # Plot reconstruction error for different num topics
    # k_list = np.arange(2, 100, 5)
    # errors = []
    # for k in k_list:
    #     nmf, W, H = fit_nmf(tfidf, n_topics=k, alpha=0.)
    #     errors.append(nmf.reconstruction_err_)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(k_list, errors)
    # ax.set_xlabel('number of topics')
    # ax.set_ylabel('reconstruction error')
    # ax.set_title('Reconstruction Error vs. Number of Topics')
    # fig.tight_layout(pad=1)
    # plt.show()
    # # fig.savefig('../images/co_NMF_reconstruction_error_vs_topics.png')
