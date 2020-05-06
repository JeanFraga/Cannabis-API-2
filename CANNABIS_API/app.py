"""
Build my app factory and do routes and configuration
"""

from decouple import config
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from .predict import predict_strain
load_dotenv()


def create_app():
    app = Flask(__name__)

    @app.route('/api/<text>', methods=['GET'])
    def predicted_strain(text=None):
        predictions = predict_strain(text)
        return jsonify(predictions)

    @app.route('/')
    def root():
        return render_template('base.html', tittle='Home')
    
    @app.route('/suggestion', methods=['POST'])
    @app.route('/suggestion/<text>', methods=['GET'])
    def suggestion(text=None, message=''):
        text = text or request.values['string']
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