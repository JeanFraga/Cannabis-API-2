"""
Build my app factory and do routes and configuration
"""

from decouple import config
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import pandas as pd
import numpy as np

from .predict import predict_strain, similar_strain
load_dotenv()

arra = pd.read_csv('CANNABIS_API/models/cannabis-strains.zip')
arra = arra.Strain

def create_app():
    app = Flask(__name__)

    # This route is an older method that returns json objects from predictions
    # can be used as api for other applications
    @app.route('/api', methods=['POST'])
    @app.route('/api/<text>', methods=['GET'])
    def predicted_strain(text=None):
        text = text or request.values['string']
        predictions = predict_strain(text)
        return jsonify(predictions.to_json())

    # home page that renders base.html
    @app.route('/')
    def root():
        return render_template('base.html', tittle='Home', strains=arra)
    
    # this route can take text from suggestion or post method to return predictions
    @app.route('/suggestion', methods=['POST'])
    @app.route('/suggestion/<text>', methods=['GET'])
    def suggestion(text=None, message=''):
        text = text or request.values['string']
        # process text from user to make prediction
        try:
            if request.method == 'POST':
                message = 'These strians may help; if not try being more specific.'
            predictions = predict_strain(text)
        except Exception as e:
            message = "Something went wrong processing {}: {}".format(text, e)
        return render_template('suggestions.html', title=text,
                               predictions=predictions.to_html(),
                               message=message,
                               strains=arra)
    
    # this route will return the strains closest to the one provided by the user based on information available in the dataframe
    @app.route('/similar', methods=['POST'])
    @app.route('/similar/<text>', methods=['GET'])
    def similar(text=None, message=''):
        text = text or request.values['string']
        # process strain user chose
        try:
            if request.method == 'POST':
                message = 'These strains are closest to the one selected'
            text_str = similar_strain(text)
            predictions = predict_strain(text_str).to_html()
        except Exception as e:
            message = "The strain {}: {} does not exist in the database".format(text, e)
            predictions = 'None'
        return render_template('suggestions.html', title=text,
                               predictions=predictions,
                               message=message,
                               strains=arra)
    return app