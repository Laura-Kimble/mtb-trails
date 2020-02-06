import numpy as np 
import pandas as pd 
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


class Featurizer(object):
    ''' Class to hold the information to feature a set of documents, including sets of stopwords,
    and whether to include bigrams and/or trigrams.
    Methods of the class include the primary method to 'featurize' a set of document strings into
    tokens using the class attributes, as well as methods to update the attributes.
      '''

    def __init__(self, first_stopwords, second_stopwords, bigrams=True, trigrams=True):
        self.first_stopwords = first_stopwords
        self.second_stopwords = second_stopwords
        self.bigrams=bigrams
        self.trigrams=trigrams

    def update_stopwords(self, stopwords_to_add, add_to='first_stopwords'):
        if add_to == 'first_stopwords':
            self.first_stopwords = self.first_stopwords.union(set(stopwords_to_add))
        else:
            self.second_stopwords = self.second_stopwords.union(set(stopwords_to_add))

    def unlist_stopwords(self, stopwords_to_remove, remove_from='first_stopwords'):
        if remove_from == 'first_stopwords':
            self.first_stopwords = self.first_stopwords.difference(set(stopwords_to_remove))
        else:
            self.second_stopwords = self.second_stopwords.difference(set(stopwords_to_remove))

    def update_ngrams(self, grams='bigrams', set_to=True):
        if grams == 'bigrams':
            self.bigrams = set_to
        else:
            self.trigrams = set_to

    def featurize(self, documents):
        tokenized_docs = documents.map(self._tokenize)
        tokens_nostops = tokenized_docs.apply(self._remove_stopwords, args=(self.first_stopwords, ))
        
        if self.bigrams:
            bigrams = self._ngramize(tokens_nostops, min_count=5)
            if self.trigrams:
                # create trigrams
                trigrams = self._ngramize(bigrams)
                new_tokens = trigrams
            else:
                new_tokens = bigrams
        else:
            new_tokens = tokens_nostops

        # lemmatize and remove stopwords again
        processed_docs = new_tokens.apply(self._preprocess)
        return processed_docs

    def _tokenize(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            result.append(token)
        return result


    def _remove_stopwords(self, text, stopwords_list):
        result = []
        for word in text:
            if word not in stopwords_list:
                result.append(word)
        return result


    def _ngramize(self, text, min_count=5):
        model = Phrases(text, min_count=min_count, threshold=2)
        phraser = Phraser(model)
        return text.map(lambda x: phraser[x])


    def _get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].lower()
        tag_dict = {'j': 'a',
                'n': 'n',
                'v': 'v',
                'r': 'r'}
        return tag_dict.get(tag, 'n')


    def _lemmatize(self, text):
        pos = self._get_wordnet_pos(text)
        return WordNetLemmatizer().lemmatize(text, pos=pos)


    def _preprocess(self, text):
        result = []
        for token in text:
            if (token not in self.second_stopwords) and (len(token) > 3):
                lem = self._lemmatize(token)
                if lem not in self.second_stopwords:
                    result.append(lem)
        return result

    def print_params(self):
        print(f'Bigrams={self.bigrams}')
        print(f'Trigrams={self.trigrams}\n')
        print(f'First set of stopwords: {self.first_stopwords}\n')
        print(f'Second set of stopwords: {self.second_stopwords}.')



def get_st_descriptions():
    st_df = pd.read_pickle('../data/st_trails_df_2')
    st_df_with_desc = st_df[st_df['description_length']>=40]
    return st_df_with_desc['description']


def make_gensim_bow(processed_docs, no_below=2, no_above=0.9, keep_n=100000, keep_tokens=None):
    
    id2word = gensim.corpora.Dictionary(processed_docs)
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
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=k, id2word=id2word, passes=5, workers=2, iterations=100)
    perplexity, coherence = get_perplexity_coherence(lda_model, bow_corpus, processed_docs, id2word)
    print(f'LDA with {k} topics: Perplexity is {perplexity:0.2} and coherence is {coherence:0.2}.')
    pprint(lda_model.print_topics())