from flask import Flask, render_template, request, jsonify
import pandas as pd 
import joblib
import sys
sys.path.insert(1, 'models')
# import item_recommender as ir
# import importlib
# importlib.reload(ir)

app = Flask(__name__)

def shorten_text_col(df, colname, num_chars=50):
    df[colname] = df[colname].map(lambda x: x[:num_chars]+'...' if len(x)>num_chars else x)
    return df

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
    if trail=='':
        return 'You must select a trail.'
    recco_names, recco_ids = recommender.get_recommendations(trail, n=5)
    trail_df = df[df['name']==trail][display_cols]
    reccos_df = df[df['id'].map(lambda x: x in recco_ids)][display_cols]
    return render_template('recommendations.html', trail_df=trail_df, reccos_df=reccos_df)


@app.route('/inputs_multi', methods=['GET'])
def inputs_multi():
    trail_names_subset = df[df['rating_rounded']>=4]['name']
    return render_template('inputs_multi.html', trail_names=trail_names_subset)


@app.route('/recommendations_multi', methods=['GET', 'POST'])
def recommendations_multi():
    liked_trails = request.form.getlist("trail_names")
    if len(liked_trails)==0:
        return 'Select at least one trail.'
    user_reccos = recommender.get_user_recommendation(liked_trails, n=10)
    reccos_df = df[df['name'].map(lambda x: x in user_reccos)][display_cols]
    return render_template('recommendations_multi.html', reccos_df=reccos_df)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True)
    

