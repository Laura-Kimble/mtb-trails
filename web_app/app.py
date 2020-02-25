from flask import Flask, render_template, request, jsonify
import joblib
import sys
sys.path.insert(1, '../src')
import item_recommender as ir
import importlib
importlib.reload(ir)

app = Flask(__name__)

recommender = joblib.load('../models/trail_recommender.joblib')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# @app.route('/inputs', methods=['GET'])
# def inputs():
#     return render_template('inputs.html')

# @app.route('/recommendations', methods=['POST'])
# def recommendations():

#     data = str(request.form['article_body'])
#     pred = str(model.predict([data])[0])
#     return render_template('predict.html', article=data, predicted=pred)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

