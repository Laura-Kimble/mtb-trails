from flask import Flask, render_template, request, jsonify
import pandas as pd 
import joblib
import sys
sys.path.insert(1, '../src')
# import item_recommender as ir
# import importlib
# importlib.reload(ir)

app = Flask(__name__)

recommender = joblib.load('../models/trail_recommender.joblib')
df = pd.read_pickle('../data/co_trails_df')

def shorten_text_col(df, colname, num_chars=50):
    df[colname] = df[colname].map(lambda x: x[:num_chars]+'...' if len(x)>num_chars else x)
    return df


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/inputs', methods=['GET'])
def inputs():
    return render_template('inputs.html', df=df)

@app.route('/recommendations', methods=['POST'])
def recommendations():
    trail = request.form['trail']
    if trail=='':
        return 'You must select a trail.'

    display_cols = ['name', 'region_name', 'difficulty', 'length', 'rating', 'description']
    recco_names, recco_ids = recommender.get_recommendations(trail, n=5)
    trail_df = df[df['name']==trail][display_cols]
    trail_df = shorten_text_col(trail_df, 'description', num_chars=200)
    reccos_df = df[df['id'].map(lambda x: x in recco_ids)][display_cols]
    reccos_df = shorten_text_col(reccos_df, 'description', num_chars=200)
    return render_template('recommendations.html', trail_df=trail_df, reccos_df=reccos_df)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True)

