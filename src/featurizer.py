import nltk
from gensim.utils import simple_preprocess
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer


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
        for token in simple_preprocess(text):
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


if __name__ == '__main__':

    added_stopwords = set(['bike', 'trail', 'mountain'])
    my_featurizer = Featurizer(STOPWORDS, STOPWORDS.union(added_stopwords), bigrams=False, trigrams=False)
    my_featurizer.print_params()
    my_featurizer.update_stopwords(['ride', 'road'])
    my_featurizer.update_ngrams(grams='bigrams', set_to=True)
    my_featurizer.print_params()