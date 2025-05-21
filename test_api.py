import requests
import json

# URL-ul API-ului tău
url = "http://localhost:5000/predict-gender"

# Numele de testat
test_names = ["Maria", "Alexandru", "Ionela", "Mihai", "Carmen", "Bogdan", "Ana", "Gabriel"]

for name in test_names:
    # Datele pentru cerere
    data = {"name": name}
    
    # Trimite cererea POST
    response = requests.post(url, json=data)
    
    # Afișează rezultatul
    if response.status_code == 200:
        result = response.json()
        print(f"{name}: {result['gender']} (probabilitate feminină: {result['female_probability']:.2f})")
    else:
        print(f"Eroare pentru {name}: {response.text}")