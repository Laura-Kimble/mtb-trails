import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from gensim.parsing.preprocessing import STOPWORDS
from mtb_sklearn_lda import make_vectorizer, transform_vectorizer, get_top_words
from st_gensim_lda import get_st_descriptions, featurize_text


def get_st_trail_names():
    st_df = pd.read_pickle('../data/st_trails_df_2')
    st_df_with_desc = st_df[st_df['description_length']>=40]
    return st_df_with_desc['name']


def get_st_regions():
    st_df = pd.read_pickle('../data/st_trails_df_2')
    st_df_with_desc = st_df[st_df['description_length']>=40]
    return st_df_with_desc['region']


def print_top_topic_words(H, vectorizer_model, n=10):
    feature_names = np.array(vectorizer_model.get_feature_names())
    for i, topic in enumerate(H):
        top_topic_words = feature_names[topic.argsort()[:-n-1:-1]]
        print(f'Topic {i}:')
        print(top_topic_words)


def print_topic_trails(W, vectorizer_model, n=10):
    trail_names = get_st_trail_names()
    regions = get_st_regions()
    for i, topic in enumerate(W.T):
        top_ind = topic.argsort()[:-n-1:-1]
        top_trails = trail_names[top_ind]
        top_regions = regions[top_ind]
        print(f'Topic {i}:')
        for trail, region in zip(top_trails, top_regions):
            print(f'{trail}: {region}')
        print('\n')


def fit_nmf(mat, n_topics, alpha=0.):
    nmf = NMF(n_components = n_topics, alpha=alpha)
    nmf.fit(mat)
    W = nmf.transform(mat)
    H = nmf.components_
    return nmf, W, H


if __name__ == '__main__':
    
    # Get singletracks trail summary data
    X = get_st_descriptions()

    # Create  set of stopwords to use 
    my_stopwords = set(['climb', 'mountain', 'road', 'singletrack', 'loop', 'trail', 'trails',  'ride', 'area', 'route', 'way', \
                        'feature', 'section','sections', 'riding', \
                    'loop','br', 'mile', 'miles', 'right', 'left', 'www', 'http', 'https', 'bike', 'bikes', 'bicycle', 'bicycles', \
                    'continue', 'rider', 'riders', 'parking', 'lot', 'turn', 'start', 'starts', 'description', 'cross', \
                    'north', 'south', 'east', 'west', '-PRON-', 'pron', 'nee', 'regard', 'shall', 'use', 'win', \
                    'park', 'point', 'biking', 'follow', 'single', 'track', 'intersection', 'trailhead', 'head', \
                    'good', 'great', 'nice', 'time', 'include', 'place', 'come', 'downhill', 'look', 'near'])
    bitri_stops = set(['parking_lot', 'trail_starts', 'mile_turn', 'north_south', 'mountain_bike', 'mountain_biking', 'single_track', \
                    'mountain_bike_trail', 'trail_head'])
    all_stopwords = my_stopwords.union(STOPWORDS).union(bitri_stops)

    # Create TF-IDF matrix
    tfidf_vect = make_vectorizer(X, tf_idf=True, lemmatize=False, max_df=0.8, min_df=2, max_features=1000, stop_words=all_stopwords, ngram_range=(1, 2))
    tfidf = transform_vectorizer(tfidf_vect, X)
    # top_words = get_top_words(tfidf_vect, tfidf, n=50)

    # NMF
    k = 6
    nmf, W, H = fit_nmf(tfidf, n_topics=k, alpha=0.)

    # Plot reconstruction error for different num topics
    # k_list = np.arange(2, 15, 2)
    # errors = []
    # for k in k_list:
    #     nmf, W, H = fit_nmf(tfidf, n_topics=k, alpha=0.)
    #     errors.append(nmf.reconstruction_err_)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(k_list, errors)
    # plt.show()


    print('reconstruction error:', nmf.reconstruction_err_)

    print_top_topic_words(H, tfidf_vect, n=10)
    print('\n\n')
    print_topic_trails(W, tfidf_vect, n=10)

