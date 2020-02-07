import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
import joblib
import spacy


def make_vectorizer(contents, tf_idf=False, lemmatize=False, **kwargs):

    if tf_idf:
        Vectorizer = TfidfVectorizer
    else:
        Vectorizer = CountVectorizer
    
    def tokenize(text):
        sp = spacy.load('en')
        lems = [word.lemma_ for word in sp(text) if word.pos_ not in ['PUNCT', 'PART', 'DET']]
        return lems

    if lemmatize:
        tokenizer = tokenize
    else:
        tokenizer=None

    vectorizer_model = Vectorizer(tokenizer=tokenizer, **kwargs)
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
    
    # Get trail summary data
    trails_df = pd.read_pickle('../data/mtb_trails_df_2')
    trails_df_with_summary = trails_df[trails_df['no_summary']==0]
    X = trails_df_with_summary['summary']

    # Update stopwords
    nltk_stopwords = set(stopwords.words('english'))
    gensim_stopwords = STOPWORDS
    my_stopwords = set(['singletrack', 'loop', 'trail', 'trails',  'ride', 'area', 'route', 'way', 'feature', 'section', 'riding'\
                    'north', 'south', 'east', 'west', '-PRON-', 'pron', 'nee', 'regard', 'shall', 'use', 'win'])
    all_stopwords = my_stopwords.union(nltk_stopwords.union(gensim_stopwords))

    # Create TF matrix
    tf_vect = make_vectorizer(X, tf_idf=False, lemmatize=False, max_df=0.8, min_df=2, max_features=1000, stop_words=all_stopwords, ngram_range=(1, 1))
    tf = transform_vectorizer(tf_vect, X)
    top_words = get_top_words(tf_vect, tf, n=50)

    # LDA
    num_topics = 3
    lda = LatentDirichletAllocation(n_components=num_topics, learning_method='online', n_jobs=-1, doc_topic_prior=0.9, topic_word_prior=0.9, random_state=64)
    lda.fit(tf)

    print_topic_words(lda, tf_vect, n=10)
    print("Model perplexity: {0:0.3f}".format(lda.perplexity(tf)))

    # PCA
    tfidf_vect = make_vectorizer(X, tf_idf=True, lemmatize=False, max_df=0.8, min_df=2, max_features=1000, stop_words=all_stopwords, ngram_range=(1, 1))
    tfidf = transform_vectorizer(tfidf_vect, X)
    pca = PCA(n_components=2)
    pca_tfidf = pca.fit_transform(tfidf.toarray())

    # Dump and reload
    # joblib.dump(lda, 'lda_model.joblib')
    # joblib.dump(tf_vect, 'tf_vec.joblib')
    # lda = joblib.load('lda_model.joblib')
    # tf_vectorizer = joblib.load('tf_vec.joblib')