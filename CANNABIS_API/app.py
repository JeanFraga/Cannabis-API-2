"""
Build my app factory and do routes and configuration
"""

from decouple import config
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from .predict import predict_strain
load_dotenv()


def create_app():
    app = Flask(__name__)

    @app.route('/<text>', methods=['GET'])
    def predicted_strain(text=None):
        predictions = predict_strain(text)
        return jsonify(predictions)

    return app