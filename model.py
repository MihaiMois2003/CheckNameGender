import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import joblib
import re
import unidecode
import random

class GenderClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        
    def preprocess_name(self, name):
        # Convertește la lowercase și elimină diacriticele
        name = name.lower()
        name = unidecode.unidecode(name)
        
        # Elimină caracterele non-alfabetice
        name = re.sub(r'[^a-z]', '', name)
        
        # Adaugă spații între caractere pentru a captura secvențe
        spaced_name = ' '.join(name)
        
        # Adaugă caracteristici pentru terminație (ultimele 1-3 caractere)
        suffix_features = []
        if len(name) >= 1:
            suffix_features.append("END1_" + name[-1:])
        if len(name) >= 2:
            suffix_features.append("END2_" + name[-2:])
        if len(name) >= 3:
            suffix_features.append("END3_" + name[-3:])
            
        # Adaugă caracteristici pentru început (primele 1-3 caractere)
        prefix_features = []
        if len(name) >= 1:
            prefix_features.append("START1_" + name[:1])
        if len(name) >= 2:
            prefix_features.append("START2_" + name[:2])
        if len(name) >= 3:
            prefix_features.append("START3_" + name[:3])
            
        # Combină toate caracteristicile
        return spaced_name + " " + " ".join(suffix_features) + " " + " ".join(prefix_features)
    
    def train(self, names, genders):
        """Antrenează modelul cu liste de nume și genuri."""
        # Preprocesare
        preprocessed_names = [self.preprocess_name(name) for name in names]
        
        # Extragere caracteristici cu n-grame
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
        X = self.vectorizer.fit_transform(preprocessed_names)
        
        # Codificarea etichetelor (1 = feminin, 0 = masculin)
        y = np.array([1 if gender == 'female' else 0 for gender in genders])
        
        # Verifică distribuția claselor
        female_count = sum(y == 1)
        male_count = sum(y == 0)
        print(f"Distribuția claselor: {female_count} feminine, {male_count} masculine")
        
        # Împărțirea datelor cu stratificare pentru a păstra distribuția claselor
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Antrenare model cu parametri optimizați
        self.model = RandomForestClassifier(
            n_estimators=200,  # Mai mulți arbori
            max_depth=10,      # Limitează adâncimea pentru a evita supraantrenarea
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',  # Compensează pentru dezechilibrul între clase
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluare
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Afișează rezultatele pentru setul de testare
        from sklearn.metrics import classification_report
        print("\nEvaluare pe setul de testare:")
        print(classification_report(y_test, y_pred, target_names=['male', 'female']))
        
        # Verifică predicțiile pentru numele din dataset
        print("\nVerificare predicții pentru unele nume din dataset:")
        sample_names = random.sample(names, min(10, len(names)))
        for name in sample_names:
            idx = names.index(name)
            gender = genders[idx]
            result = self.predict(name)
            correct = (result['gender'] == gender)
            print(f"{name}: Prezis {result['gender']}, Actual {gender}, Corect: {correct}, "
                  f"Probabilitate feminină: {result['female_probability']:.2f}")
        
        return accuracy
    
    def predict(self, name):
        """Prezice genul și probabilitatea că este feminin."""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Modelul nu a fost antrenat!")
            
        # Extrage caracteristici
        features = self.preprocess_name(name)
        X = self.vectorizer.transform([features])
        
        # Prezice probabilitățile
        proba = self.model.predict_proba(X)[0]
        
        # Prezice clasa (0 = masculin, 1 = feminin)
        gender = 'female' if proba[1] > 0.5 else 'male'
        
        return {
            'name': name,
            'gender': gender,
            'female_probability': float(proba[1]),
            'is_female': gender == 'female'
        }
    
    def save(self, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
        """Salvează modelul și vectorizatorul."""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Modelul nu a fost antrenat!")
            
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
    def load(self, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
        """Încarcă modelul și vectorizatorul."""
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)