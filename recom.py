import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

classes = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
       'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
       'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
       'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

clf = pickle.load(open('recommender.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [request.form['a'], request.form['b'], request.form['c'], request.form['d'], request.form['e'], request.form['f'], request.form['g']]
        for i in range(len(features)):
             features[i] = float(features[i])
	# features = eval(request.form.values())
        print(features)
        pred = clf.predict([features])

    return jsonify({'crop' : classes[pred[0]]})

if __name__ == "__main__":
    app.run(port = 9000)