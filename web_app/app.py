from flask import Flask, render_template, request
import pandas as pd 
import numpy as np 
import joblib


app = Flask(__name__)

def shorten_text_col(df, colname, num_chars=50):
    df[colname] = df[colname].map(lambda x: x[:num_chars]+'...' if len(x)>num_chars else x)
    return df


def filter_on_distance(recco_ids, max_distance, df):
    distance_buckets = ['0-25 miles', '25-50 miles', '50-100 miles', '100-200 miles', '200+ miles']
    acceptable_buckets = distance_buckets[0:distance_buckets.index(max_distance) + 1]
    filtered_df = df[df['id'].map(lambda x: x in recco_ids)]
    filtered_df = filtered_df[filtered_df['dist_to_Denver_mi_bucket'].map(lambda x: x in acceptable_buckets)]
    filtered_ids = np.array(filtered_df['id'])
    return list(filtered_ids)


recommender = joblib.load('models/trail_recommender.joblib')
df = pd.read_pickle('data/co_trails_df')
df = shorten_text_col(df, 'description', num_chars=200)
df = shorten_text_col(df, 'description', num_chars=200)
display_cols = ['name', 'url', 'region_name', 'difficulty', 'length', 'rating', 'description']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/inputs', methods=['GET'])
def inputs():
    trail_names = recommender.trail_names
    return render_template('inputs.html', trail_names=trail_names)


@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    trail = request.form['trail']
    max_distance = request.form['maxDist']
    if trail=='':
        return 'You must select a trail.'
    recco_names, recco_ids = recommender.get_recommendations(trail, n=5)
    filtered_ids = filter_on_distance(recco_ids, max_distance, df)
    trail_df = df[df['name']==trail][display_cols]
    reccos_df = df[df['id'].map(lambda x: x in filtered_ids)][display_cols]
    return render_template('recommendations.html', trail_df=trail_df, reccos_df=reccos_df)


@app.route('/inputs_multi', methods=['GET'])
def inputs_multi():
    trail_names_subset = df[(df['rating_rounded'] >= 4) & (df['description_length'] >= 40)]['name']
    return render_template('inputs_multi.html', trail_names=trail_names_subset)


@app.route('/recommendations_multi', methods=['GET', 'POST'])
def recommendations_multi():
    liked_trails = request.form.getlist("trail_names")
    if len(liked_trails)==0:
        return 'Select at least one trail.'
    user_reccos = recommender.get_user_recommendation(liked_trails, n=10)
    reccos_df = df[df['name'].map(lambda x: x in user_reccos)][display_cols]
    return render_template('recommendations_multi.html', reccos_df=reccos_df)


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True)
    

