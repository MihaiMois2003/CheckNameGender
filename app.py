from flask import Flask, request, jsonify
from flask_cors import CORS
from model import GenderClassifier
import os

app = Flask(__name__)
CORS(app)  # Permite cereri CORS din aplicația React

# Inițializează modelul
classifier = GenderClassifier()

# Încarcă modelul dacă există, altfel antrenează un model nou
if os.path.exists('model.joblib') and os.path.exists('vectorizer.joblib'):
    classifier.load('model.joblib', 'vectorizer.joblib')
    print("Model încărcat cu succes!")
else:
    print("Nu s-a găsit modelul antrenat. Executați mai întâi train.py!")

@app.route('/predict-gender', methods=['POST'])
def predict_gender():
    data = request.get_json()
    
    if not data or 'name' not in data:
        return jsonify({'error': 'Numele lipsește'}), 400
    
    name = data['name']
    
    try:
        result = classifier.predict(name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)