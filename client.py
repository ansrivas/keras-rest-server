# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Client to send inputs for predictions."""

import requests


def get_predictions(X_input):
    """Get predictions from a rest backend for your input."""
    print("Requesting prediction for XOR with {0}".format(X_input))
    r = requests.post("http://localhost:7171/predict", json={'X_input': X_input})
    print(r.status_code, r.reason)
    resp = r.json()
    prediction = resp['pred_val'][0]
    print("XOR of input: {0} is {1} ".format(X_input, prediction))


if __name__ == '__main__':

    X_inputs = [[1., 1.], [1., 0.]]

    for x_input in X_inputs:
        get_predictions(x_input)
