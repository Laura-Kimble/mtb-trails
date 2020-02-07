import numpy as np 
import pandas as pd 
from gensim.models import LdaMulticore
import gensim.corpora as corpora
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import CoherenceModel
from pprint import pprint
from featurizer import Featurizer


def get_st_descriptions():
    st_df = pd.read_pickle('../data/st_trails_df_2')
    st_df_with_desc = st_df[st_df['description_length']>=40]
    return st_df_with_desc['description']


def make_gensim_bow(processed_docs, no_below=2, no_above=0.9, keep_n=100000, keep_tokens=None):
    
    id2word = corpora.Dictionary(processed_docs)
    id2word.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n, keep_tokens=keep_tokens)
    
    bow_corpus = [id2word.doc2bow(text) for text in processed_docs]
    return bow_corpus, id2word


def get_perplexity_coherence(lda_model, bow_corpus, processed_docs, id2word, coherence='c_v'):
    
    perplexity = lda_model.log_perplexity(bow_corpus)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    return perplexity, coherence_lda


if __name__ == '__main__':
    
    # Get singltracks trail summary data
    X = get_st_descriptions()

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
    st_featurizer = Featurizer(first_stopwords=first_stopwords, second_stopwords=second_stopwords, bigrams=True, trigrams=True)
    processed_docs = st_featurizer.featurize(X)
    bow_corpus, id2word = make_gensim_bow(processed_docs, no_below=3, no_above=0.6, keep_n=10000)

    k = 6
    lda_model = LdaMulticore(bow_corpus, num_topics=k, id2word=id2word, passes=5, workers=2, iterations=100)
    perplexity, coherence = get_perplexity_coherence(lda_model, bow_corpus, processed_docs, id2word)
    print(f'LDA with {k} topics: Perplexity is {perplexity:0.2} and coherence is {coherence:0.2}.')
    pprint(lda_model.print_topics())