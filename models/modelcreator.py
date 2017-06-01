# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Base module to create the model from its normalization coefficients."""

from keras.models import model_from_json

try:
    import cPickle as pickle
except Exception as ex:
    import pickle


class ModelOperations(object):
    """ModelOperations class deals with saving/loading the model weights from Keras models."""

    def __init__(self):
        """Initialize ModelOperations class."""
        pass

    def load_model(self, json_path, weights_path):
        """Load model from a given json path and weights path."""
        try:

            model = model_from_json(open(json_path).read())
            model.load_weights(weights_path)
            return model
        except Exception as ex:
            raise Exception('Failed to load model/weights')

    def load_normalizer(self, sk_normalized):
        """load_normalizer will loads the sklearn.preprocessing.StandardScaler object.

        This had been used to normalize the original dataset.
        """
        try:
            f = open(sk_normalized, 'rb')
            scalar = pickle.load(f)
            f.close()
            return scalar
        except Exception as ex:
            raise Exception('Failed to load normalizer')

    def save_model(self, model, json_path, weights_path):
        """Helper wrapper over savemodels and saveweights to help keras dump the weights and configuration."""
        json_string = model.to_json()
        with open(json_path, 'w') as f:
            f.write(json_string)
        model.save_weights(weights_path)


class Predictor(object):
    """Predictor class does the heavy lifting of compiling Keras model, normalizing inputs, denormalizing outputs."""

    def __init__(self, json_path, weights_path, normalized_x, normalized_y, **kwargs):
        """Initialize Predictor class."""
        modoperations = ModelOperations()
        self.model = modoperations.load_model(json_path, weights_path)
        self.scalar_x = modoperations.load_normalizer(normalized_x)
        self.scalar_y = modoperations.load_normalizer(normalized_y)

    def compile_model(self, loss, optimizer, **kwargs):
        """Similar to Keras compile function, expects atleast losstype and optimizer."""
        self.model.compile(loss=loss, optimizer=optimizer, **kwargs)

    def _normalize_input(self, X_input):
        """Normalize the input object to be predicted according to the scalar used during the training process."""
        X_input = self.scalar_x.transform(X_input)
        return X_input

    def _denormalize_prediction(self, x_pred):
        """De-normalize the x_pred to actual value as per dataset."""
        value = self.scalar_y.inverse_transform(x_pred)
        return value

    def predict(self, X_input):
        """Make predictions, given some input data.

        This normalizes the predictions based on the real normalization
        parameters and then generates a prediction

        Args:
            X_input:    Input vector to for prediction
        """
        x_normed = self._normalize_input(X_input=X_input)
        x_pred = self.model.predict(x_normed)
        prediction = self._denormalize_prediction(x_pred)
        return prediction
