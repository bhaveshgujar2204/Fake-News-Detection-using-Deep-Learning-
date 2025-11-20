import requests

url = "http://127.0.0.1:5000/predict"
data = {"text": "Sun is a Planet"}
response = requests.post(url, json=data)

print(response.json())  # Should return {"prediction": "REAL"} or {"prediction": "FAKE"}
