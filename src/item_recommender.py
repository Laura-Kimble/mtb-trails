from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


class ItemRecommender():
    '''
    Content based item recommender
    '''
    def __init__(self, similarity_measure=None):
        self.similarity_measure = similarity_measure

    
    def fit(self, X, trail_names, trail_ids):
        '''
        Takes a numpy array of the item attributes and creates similarity matrix

        INPUT -
            X: NUMPY ARRAY - Rows are items, columns are feature values
            trail_names: LIST of the item names/titles in order of the numpy arrray
            trail_ids: LIST of the trail ids to look up in the oringal dataframe with add'l trail data
        
        OUTPUT - None

        Create the a similarity matrix of item to item similarity
        '''
        self.X = X
        self.trail_names = trail_names
        self.trail_ids = trail_ids

        if self.similarity_measure:
            similarity_matrix = self.similarity_measure(X)
        else:  # if similarity measure not specified (None), use cosine similarity
            similarity_matrix = cosine_similarity(X)
        self.similarity_matrix = similarity_matrix

        
    def get_recommendations(self, item, n=5):
        '''
        Returns the top n items related to the item passed in.

        INPUT:
            item    - STRING - Name of item in the original DataFrame 
            n       - INT    - Number of top related items to return 
        OUTPUT:
            items, item_ids - Lists of the top n related item names, and the similar item ids
        '''

        if item not in self.trail_names:
            print('Item not in data set')
            return None

        item_index = np.argwhere(np.array(self.trail_names) == item)[0][0]
        item_row = self.similarity_matrix[item_index]
        similar_ind = np.argsort(item_row)[-2:-n - 2:-1]
        similar_items = np.array(self.trail_names)[similar_ind]
        similar_item_ids = np.array(self.trail_ids)[similar_ind]
        return list(similar_items), list(similar_item_ids)


    def get_user_profile(self, items):
        '''
        Takes a list of items and returns a user profile: A vector representing the likes of the user.
        
        INPUT: 
            items  -   LIST - list of trail names the user likes/has ridden

        OUTPUT: 
            user_profile - NP ARRAY - array representing the likes of the user 
                    The columns of this will match the columns of the trained on matrix
        '''
        item_indices = []
        for item in items:
            item_ind = np.argwhere(np.array(self.trail_names) == item)[0][0]
            item_indices.append(item_ind)
        user_profile = np.sum(self.X.iloc[item_indices], axis=0)
        user_profile = np.array(user_profile)
        return user_profile


    def get_user_recommendation(self, items, n=5):
        '''
        Takes a list of trails user liked and returns the top n items for that user

        INPUT 
            items  -   LIST - list of trail names user likes / has ridden
            n -  INT - number of items to return

        OUTPUT 
            results - LIST - n recommended items
        '''
        user_profile = self.get_user_profile(items)
        y = np.repeat(user_profile.reshape(1, - 1), self.X.shape[0], axis=0) 
        similarities = self.similarity_measure(self.X, y)[:, 0]
        results = []
        for i in range(1, len(self.trail_names)+1):
            if len(results) == n:
                return results
            similar_ind = np.argsort(similarities)[:-i-1: -1][-1]
            similar_trail = self.trail_names[similar_ind]
            if similar_trail not in items:
                results.append(similar_trail)
        return results

                







