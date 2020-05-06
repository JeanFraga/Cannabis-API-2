"""
Build my app factory and do routes and configuration
"""

from decouple import config
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from .predict import predict_strain
load_dotenv()

# app factory method

def create_app():
    app = Flask(__name__)

    # This route is an older method that returns json objects from predictions
    # can be used as api for other applications
    @app.route('/api/<text>', methods=['GET'])
    def predicted_strain(text=None):
        predictions = predict_strain(text)
        return jsonify(predictions.to_json())

    # home page that renders base.html
    @app.route('/')
    def root():
        return render_template('base.html', tittle='Home')
    
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
                               message=message)

    return app