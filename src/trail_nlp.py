import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
import joblib
from sklearn.metrics.pairwise import cosine_distances



if __name__ == '__main__':
    
    trails_df = pd.read_pickle('../data/mtb_trails_df_2')
    trails_df_with_summary = trails_df[trails_df['no_summary']==0]
    X = trails_df_with_summary['summary']

    my_stopwords = set(['trail', 'ride', 'trails', 'area', 'route', 'way', 'features', 'section', 'doubletrack', 'riding'\
                        'section', 'sections', 'north', 'south', 'east', 'west'])
    nltk_stopwords = set(stopwords.words('english'))
    gensim_stopwords = STOPWORDS
    all_stopwords = my_stopwords.union(nltk_stopwords.union(gensim_stopwords))

    tf_vect = CountVectorizer(max_df=1.0, min_df=2, max_features=1000, stop_words=all_stopwords, ngram_range=(1, 1))
    tf = tf_vect.fit_transform(X)
    feature_names = np.array(tf_vect.get_feature_names())
    word_freq = np.array(tf.sum(axis=0)).flatten()

    n = 50
    top_words = feature_names[word_freq.argsort()[:-n:-1]]
    

    # stem / lemmetize
    # remove stop words(trail, ride)

    # PCA w/ 2 dimensions to visualize?
    # n-grams