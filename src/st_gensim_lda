import numpy as np 
import pandas as pd 
import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.metrics.pairwise import cosine_distances

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import CoherenceModel, Phrases
from gensim.models.phrases import Phraser

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from pprint import pprint


def featurize_text(documents, first_stopwords, second_stopwords, bigrams=True, trigrams=True):
    ''' Take a set of documents as sentences and process them into tokens, remove first set of stop words, 
    create n-grams w/o those stops, remove additional stop words, lemmatize,
    and return a document series where each doc is a list of n-gram tokens.
    '''
    tokenized_docs = documents.map(tokenize)
    tokens_nostops = tokenized_docs.apply(remove_stopwords, args=(first_stopwords, ))
    
    if bigrams:
        bigrams = ngramize(tokens_nostops, min_count=5)
        if trigrams:
            # create trigrams
            trigrams = ngramize(bigrams)
            new_tokens = trigrams
        else:
            new_tokens = bigrams
    else:
        new_tokens = tokens_nostops

    # remove second set of stopwords
    # new_tokens_nostops = new_tokens.apply(remove_stopwords, args=(second_stopwords, ))
    # lemmatize and remove stopwords again
    processed_docs = new_tokens.apply(preprocess, args=(second_stopwords, ))
    return processed_docs


def tokenize(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        result.append(token)
    return result


def remove_stopwords(text, stopwords_list):
    result = []
    for word in text:
        if word not in stopwords_list:
            result.append(word)
    return result


def ngramize(text, min_count=5):
    model = Phrases(text, min_count=min_count, threshold=2)
    phraser = Phraser(model)
    return text.map(lambda x: phraser[x])


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].lower()
    tag_dict = {'j': 'a',
            'n': 'n',
            'v': 'v',
            'r': 'r'}
    return tag_dict.get(tag, 'n')


def lemmatize_(text):
    pos = get_wordnet_pos(text)
    return WordNetLemmatizer().lemmatize(text, pos=pos)

def preprocess(text, stopwords_list):
    result = []
    for token in text:
        if (token not in stopwords_list) and (len(token) > 3):
            lem = lemmatize_(token)
            if lem not in stopwords_list:
                result.append(lem)
    return result


def make_gensim_bow(processed_docs, no_below=2, no_above=0.9, keep_n=100000, keep_tokens=None):

    id2word = gensim.corpora.Dictionary(processed_docs)
    id2word.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n, keep_tokens=keep_tokens)

    bow_corpus = [id2word.doc2bow(text) for text in processed_docs]
    return bow_corpus, id2word


def get_perplexity_coherence(lda_model, bow_corpus, processed_docs, id2word, coherence='c_v'):
    perplexity = lda_model.log_perplexity(bow_corpus) # a measure of how good the model is. lower the better.
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    return perplexity, coherence_lda


# def make_vectorizer(contents, tf_idf=False, lemmatize=False, **kwargs):

#     if tf_idf:
#         Vectorizer = TfidfVectorizer
#     else:
#         Vectorizer = CountVectorizer
    
#     def tokenize(text):
#         sp = spacy.load('en')
#         lems = [word.lemma_ for word in sp(text) if word.pos_ not in ['PUNCT', 'PART', 'DET']]
#         return lems

#     if lemmatize:
#         tokenizer = tokenize
#     else:
#         tokenizer=None

#     vectorizer_model = Vectorizer(tokenizer=tokenizer, **kwargs)
#     vectorizer_model.fit(contents)
#     return vectorizer_model


# def transform_vectorizer(vectorizer_model, contents):
#     return vectorizer_model.transform(contents)


# def get_top_words(vectorizer_model, tf_matrix, n=20):
#     ''' Given a tf or tf-idf vectorizer model and an array of text,
#     return a dictionary where the top n words in the corpus are keys and the frequency of each word is the value.
#     '''
#     feature_names = np.array(vectorizer_model.get_feature_names())
#     word_freq = np.array(tf_matrix.sum(axis=0)).flatten()
#     top_indices = word_freq.argsort()[:-n-1:-1]
#     top_words = feature_names[top_indices]
#     top_freq = word_freq[top_indices]
#     return dict(zip(top_words, top_freq))

# def print_topic_words(lda_model, vectorizer_model, n=10):
#     feature_names = np.array(vectorizer_model.get_feature_names())
#     phi = lda_model.components_
#     for i, topic in enumerate(phi):
#         top_topic_words = feature_names[topic.argsort()[:-n-1:-1]]
#         print(f'Topic {i}:')
#         print(top_topic_words)
    

if __name__ == '__main__':
    
    # Get trail summary data
    st_df = pd.read_pickle('../data/st_trails_df_2')
    st_df_with_desc = st_df[st_df['description_length']>=40]
    X = st_df_with_desc['description']

    # Create initial stopwords to remove before creating n-grams
    not_stops_firstpass = ['not', 'bottom', 'few', 'many', 'more', 'less', 'most', 'least', 'never', 'off', 'out'\
                            'very', 'too', 'overly', 'so']
    new_stops_firstpass = ['br']
    first_stopwords = (STOPWORDS.difference(not_stops_firstpass)).union(new_stops_firstpass)

    # Create second set of stopwords to use after creating n-grams
    my_stopwords = set(['climb', 'mountain', 'road', 'singletrack', 'loop', 'trail', 'trails',  'ride', 'area', 'route', 'way', \
                        'feature', 'section','sections', 'riding', \
                    'loop','br', 'mile', 'miles', 'right', 'left', 'www', 'http', 'https', 'bike', 'bikes', 'bicycle', 'bicycles', \
                    'continue', 'rider', 'riders', 'parking', 'lot', 'turn', 'start', 'starts', 'description', 'cross', \
                    'north', 'south', 'east', 'west', '-PRON-', 'pron', 'nee', 'regard', 'shall', 'use', 'win', \
                    'park', 'point', 'biking', 'follow', 'single', 'track', 'intersection', 'trailhead', 'head', \
                    'good', 'great', 'nice', 'time', 'include', 'place', 'come', 'downhill', 'look', 'near'])
    bitri_stops = set(['parking_lot', 'trail_starts', 'mile_turn', 'north_south', 'mountain_bike', 'mountain_biking', 'single_track', \
                    'mountain_bike_trail', 'trail_head'])
    second_stopwords = my_stopwords.union(STOPWORDS).union(bitri_stops)

    # Gensim LDA
    processed_docs = featurize_text(X, first_stopwords=first_stopwords, second_stopwords=second_stopwords, bigrams=True, trigrams=True)
    bow_corpus, id2word = make_gensim_bow(processed_docs, no_below=3, no_above=0.6, keep_n=10000)

    k = 6
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=k, id2word=id2word, passes=5, workers=2, iterations=100)
    perplexity, coherence = get_perplexity_coherence(lda_model, bow_corpus, processed_docs, id2word)
    print(f'LDA with {k} topics: Perplexity is {perplexity:0.2} and coherence is {coherence:0.2}.')
    pprint(lda_model.print_topics())



    # # Create TF matrix
    # tf_vect = make_vectorizer(X, tf_idf=False, lemmatize=False, max_df=0.8, min_df=2, max_features=1000, stop_words=all_stopwords, ngram_range=(1, 1))
    # tf = transform_vectorizer(tf_vect, X)
    # top_words = get_top_words(tf_vect, tf, n=50)

    # LDA
    # num_topics = 8
    # lda = LatentDirichletAllocation(n_components=num_topics, learning_method='online', n_jobs=-1, doc_topic_prior=0.9, topic_word_prior=0.9, random_state=64)
    # lda.fit(tf)

    # print_topic_words(lda, tf_vect, n=10)
    # print("Model perplexity: {0:0.3f}".format(lda.perplexity(tf)))

    # PCA
    # tfidf_vect = make_vectorizer(X, tf_idf=True, lemmatize=False, max_df=0.8, min_df=2, max_features=1000, stop_words=all_stopwords, ngram_range=(1, 1))
    # tfidf = transform_vectorizer(tfidf_vect, X)
    # pca = PCA(n_components=2)
    # pca_tfidf = pca.fit_transform(tfidf.toarray())




    # Dump and reload
    # joblib.dump(lda, 'lda_model.joblib')
    # joblib.dump(tf_vect, 'tf_vec.joblib')
    # lda = joblib.load('lda_model.joblib')
    # tf_vectorizer = joblib.load('tf_vec.joblib')
    


    # PCA w/ 2 dimensions to visualize?
    # num topics
    # n-grams
    # NMF
    # doc priors
    # gensim viz
    # scoring