import requests

x = [1., 1.]
r = requests.post("http://localhost:7171/predict", json={'X_input': x})
print(r.status_code, r.reason)
resp = r.json()
print resp['pred_val'][0]
