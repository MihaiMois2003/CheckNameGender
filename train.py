import pandas as pd
from model import GenderClassifier

def train_model():
    print("Încărcăm datele...")
    try:
        # Încarcă setul de date extins
        data = pd.read_csv('names_extended.csv')
        print("Coloanele din CSV:", data.columns.tolist())
        print("Primele 5 rânduri din CSV:")
        print(data.head())
        
        print(f"Dataset încărcat cu {len(data)} nume.")
        
        # Inițializează și antrenează modelul
        classifier = GenderClassifier()
        accuracy = classifier.train(data['name'].tolist(), data['gender'].tolist())
        
        print(f"Model antrenat cu acuratețe: {accuracy * 100:.2f}%")
        
        # Salvează modelul
        classifier.save('model.joblib', 'vectorizer.joblib')
        print("Model salvat cu succes!")
        
        # Testează modelul cu nume specifice
        test_names = ['Maria', 'Alexandru', 'Ionela', 'Mihai', 'Carmen', 'Bogdan', 'Ana', 'Gabriel']
        print("\nTestare finală a modelului:")
        for name in test_names:
            result = classifier.predict(name)
            print(f"{name}: {result['gender']} (probabilitate feminină: {result['female_probability']:.2f})")
    
    except Exception as e:
        print(f"Eroare la încărcarea sau procesarea datelor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model()