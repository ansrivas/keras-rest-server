# Keras-rest-server: A simple rest implementation for loading and serving keras models
------------------
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
