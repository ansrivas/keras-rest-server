# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model server with some hardcoded paths."""

from flask.views import MethodView

from flask import Flask, request, jsonify
from .modelcreator import Predictor
from gevent.pywsgi import WSGIServer
import numpy as np

app = Flask(__name__)
predictor = None


class ModelLoader(MethodView):
    """ModelLoader class initialzes the model params and waits for a post request to server predictions."""

    def __init__(self):
        """Initialize ModelLoader class."""
        pass

    def post(self):
        """Accept a post request to serve predictions."""
        content = request.get_json()
        X_input = content['X_input']
        if not isinstance(X_input, np.ndarray):
            X_in = np.reshape(np.array(X_input), newshape=(1, 2))
        pred_val = predictor.predict(X_input=X_in)
        pred_val = pred_val.tolist()
        return jsonify({'pred_val': pred_val})


def initialize_models(json_path, weights_path, normalized_x, normalized_y):
    """Initialize models and use this in Flask server."""
    global predictor
    predictor = Predictor(json_path, weights_path, normalized_x, normalized_y)
    predictor.compile_model(loss='mse', optimizer='rmsprop')


def run(host='0.0.0.0', port=7171):
    """Run a WSGI server using gevent."""
    app.add_url_rule('/predict', view_func=ModelLoader.as_view('predict'))
    print('running server http://{0}'.format(host + ':' + str(port)))
    WSGIServer((host, port), app).serve_forever()
