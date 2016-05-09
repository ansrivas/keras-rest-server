## Keras-rest-server: A simple rest implementation for loading and serving keras models
------------------
## About:
This repository contains a very simple server implemented in flask which loads a
a simple neural network model trained using Keras from its saved-weights and
model.

In this example a very simple case of XOR is considered.
## Getting started:

### Clone this repository
```
git clone https://github.com/ansrivas/keras-rest-server.git
cd keras-rest-server
sudo python install -r requirements.txt
```
------------------
### Run to generate pickle files:
```
python createpickles.py
```
------------------
### Run the server (defaults to http://localhost:7171)
```
python server.py
```

### Send a post request to this server to test your model
```
python client.py
```
