from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os
import json
from feature_extractor import FeatureExtractor

class ModelTrainer:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.feature_extractor = FeatureExtractor()
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        self.pipeline = Pipeline([
            ('feature_extractor', self.feature_extractor),
            ('classifier', self.model)
        ])
        
        # Load supported languages
        with open('languages.json') as f:
            self.languages = json.load(f)

    def train(self, texts, labels):
        """Train the model on the given texts and labels"""
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts):
        """Predict language for given texts"""
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        """Get probability estimates for each language"""
        return self.pipeline.predict_proba(texts)

    def save(self):
        """Save the trained model and feature extractor"""
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.pipeline, os.path.join(self.model_dir, 'model.joblib'))
        joblib.dump(self.languages, os.path.join(self.model_dir, 'languages.joblib'))

    @classmethod
    def load(cls, model_dir='models'):
        """Load a trained model from disk"""
        trainer = cls(model_dir)
        trainer.pipeline = joblib.load(os.path.join(model_dir, 'model.joblib'))
        trainer.languages = joblib.load(os.path.join(model_dir, 'languages.joblib'))
        return trainer 