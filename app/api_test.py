import requests
import json

url = 'http://127.0.0.1:8000/predict'
data = {
    "date": "03-05-2023",
    "otv_orani": 10.0,
    "faiz": 5.0,
    "euro_tl": 2.0,
    "kredi_stok": 1000.0
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", json.dumps(response.json(), indent=4))