import numpy as np 
import pandas as pd 
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import joblib
from sklearn.metrics.pairwise import cosine_distances
import spacy


def make_count_vectorizor(contents, max_df=1.0, min_df=1, max_features=1000, stop_words='english', ngram_range=(1, 1)):

    def tokenize(text):
        sp = spacy.load('en')
        lems = [word.lemma_ for word in sp(text) if word not in string.punctuation]
        return lems

    vectorizer_model = CountVectorizer(contents, tokenizer=tokenize, max_features=max_features)





    

if __name__ == '__main__':
    
    trails_df = pd.read_pickle('../data/mtb_trails_df_2')
    trails_df_with_summary = trails_df[trails_df['no_summary']==0]
    X = trails_df_with_summary['summary']

    # Lemmatize with spacy
    sp = spacy.load('en')
    for row in X

    my_stopwords = set(['trail', 'ride', 'area', 'route', 'way', 'feature', 'section', 'ride', 'riding'\
                        'north', 'south', 'east', 'west'])
    nltk_stopwords = set(stopwords.words('english'))
    gensim_stopwords = STOPWORDS
    all_stopwords = my_stopwords.union(nltk_stopwords.union(gensim_stopwords))

    sp = spacy.load('en')


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