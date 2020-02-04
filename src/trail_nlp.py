import numpy as np 
import pandas as pd 
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import joblib
from sklearn.metrics.pairwise import cosine_distances
import spacy


def make_count_vectorizer(contents, max_df=1.0, min_df=1, max_features=1000, stop_words='english', ngram_range=(1, 1)):

    def tokenize(text):
        sp = spacy.load('en')
        lems = [word.lemma_ for word in sp(text) if word.pos_ not in ['PUNCT', 'PART', 'DET']]
        return lems

    vectorizer_model = CountVectorizer(max_df=max_df, tokenizer=tokenize, min_df=min_df, max_features=max_features, stop_words=stop_words, ngram_range=ngram_range)
    vectorizer_model.fit(contents)
    return vectorizer_model


def transform_vectorizer(vectorizer_model, contents):
    return vectorizer_model.transform(contents)


def get_top_words(vectorizer_model, tf_matrix, n=20):
    ''' Given a tf or tf-idf vectorizer model and an array of text,
    return a dictionary where the top n words in the corpus are keys and the frequency of each word is the value.
    '''
    feature_names = np.array(vectorizer_model.get_feature_names())
    word_freq = np.array(tf_matrix.sum(axis=0)).flatten()
    top_indices = word_freq.argsort()[:-n-1:-1]
    top_words = feature_names[top_indices]
    top_freq = word_freq[top_indices]
    return dict(zip(top_words, top_freq))

def print_topic_words(lda_model, vectorizer_model, n=10):
    feature_names = np.array(vectorizer_model.get_feature_names())
    phi = lda_model.components_
    for i, topic in enumerate(phi):
        top_topic_words = feature_names[topic.argsort()[:-n-1:-1]]
        print(f'Topic {i}:')
        print(top_topic_words)
    

if __name__ == '__main__':
    
    trails_df = pd.read_pickle('../data/mtb_trails_df_2')
    trails_df_with_summary = trails_df[trails_df['no_summary']==0]
    X = trails_df_with_summary['summary']

    nltk_stopwords = set(stopwords.words('english'))
    gensim_stopwords = STOPWORDS
    my_stopwords = set(['trail', 'ride', 'area', 'route', 'way', 'feature', 'section', 'ride', 'riding'\
                    'north', 'south', 'east', 'west', '-PRON-', 'nee', 'regard', 'shall', 'use', 'win'])
    all_stopwords = my_stopwords.union(nltk_stopwords.union(gensim_stopwords))

    tf_vect = make_count_vectorizer(X, max_df=0.9, min_df=2, max_features=1000, stop_words=all_stopwords, ngram_range=(1, 1))
    tf = transform_vectorizer(tf_vect, X)
    top_words = get_top_words(tf_vect, tf, n=50)

    num_topics = 10
    lda = LatentDirichletAllocation(n_components=num_topics, learning_method='online', n_jobs=-1, doc_topic_prior=None, topic_word_prior=None)
    lda.fit(tf)

    print_topic_words(lda, tf_vect, n=10)

    # Dump and reload
    joblib.dump(lda, 'lda_model.joblib')
    joblib.dump(tf_vect, 'tf_vec.joblib')
    # lda = joblib.load('lda_model.joblib')
    # tf_vectorizer = joblib.load('tf_vec.joblib')
    


    # PCA w/ 2 dimensions to visualize?
    # n-grams