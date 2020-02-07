import numpy as np 
import pandas as pd 
import joblib
from sklearn.decomposition import NMF
from gensim.parsing.preprocessing import STOPWORDS
from mtb_sklearn_lda import make_vectorizer, transform_vectorizer, get_top_words
from st_gensim_lda import get_st_descriptions
import matplotlib.pyplot as plt


def get_st_trail_names():
    st_df = pd.read_pickle('../data/st_trails_df_2')
    st_df_with_desc = st_df[st_df['description_length']>=40]
    return st_df_with_desc['name']


def get_st_regions():
    st_df = pd.read_pickle('../data/st_trails_df_2')
    st_df_with_desc = st_df[st_df['description_length']>=40]
    return st_df_with_desc['region']


def print_top_topic_words(H, feature_names, n=10):
    for i, topic in enumerate(H):
        top_topic_words = feature_names[topic.argsort()[:-n-1:-1]]
        print(f'Topic {i}:')
        print(top_topic_words)


def print_topic_trails(trail_list):
    for i, trails in enumerate(trail_list):
        print(f'Topic {i}:')
        for trail, region in trails:
            print(f'{trail}: {region}')
        print('\n')


def get_topic_trails(W, n=10):
    all_topic_trails = []
    trail_names = get_st_trail_names()
    regions = get_st_regions()
    
    for topic in W.T:
        top_ind = topic.argsort()[:-n-1:-1]
        top_trails = trail_names[top_ind]
        top_regions = regions[top_ind] 
        all_topic_trails.append(list(zip(top_trails, top_regions)))
    return all_topic_trails


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
    tfidf_vect = make_vectorizer(X, tf_idf=True, lemmatize=False, max_df=0.6, min_df=2, max_features=1000, stop_words=all_stopwords, ngram_range=(1, 2))
    tfidf = transform_vectorizer(tfidf_vect, X)
    # top_words = get_top_words(tfidf_vect, tfidf, n=50)
    
    feature_names = np.array(tfidf_vect.get_feature_names())
    trail_names = get_st_trail_names()

    # NMF
    k = 6
    topics = ['topic_{}'.format(i) for i in range(k)]

    nmf, W, H = fit_nmf(tfidf, n_topics=k, alpha=0.)
    W_df = pd.DataFrame(W, index=trail_names, columns=topics)
    H_df = pd.DataFrame(H, index=topics, columns=feature_names)
    print(f'reconstruction error: {nmf.reconstruction_err_:0.3}')

    print_top_topic_words(H, feature_names, n=10)
    print('\n\n')
    all_topic_trails = get_topic_trails(W, n=10)
    print_topic_trails(all_topic_trails)

    # Dump vectorizer, model and W and H as dataframes
    joblib.dump(nmf, '../models/nmf_model.joblib')
    joblib.dump(tfidf_vect, '../models/tfidf_vec.joblib')

    W_df = pd.DataFrame(W, index=trail_names, columns=topics)
    H_df = pd.DataFrame(H, index=topics, columns=feature_names)
    W_df.to_pickle('../models/W_df')
    H_df.to_pickle('../models/H_df')

    # Plot reconstruction error for different num topics
    # k_list = np.arange(2, 15, 2)
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
    # # plt.show()
    # fig.savefig('../images/NMF_reconstruction_error_vs_topics.png')

