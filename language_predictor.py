import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import time
import chardet
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from collections import Counter
from sklearn.metrics import accuracy_score

class LanguagePredictor:
    """Language prediction system with backward compatibility"""
    
    def __init__(self, model_dir: str = 'models', use_legacy: bool = False):
        """
        Initialize the language predictor
        
        Args:
            model_dir: Directory to store/load models
            use_legacy: Whether to use the legacy TensorFlow model
        """
        self.model_dir = model_dir
        self.use_legacy = use_legacy
        
        # Load supported languages
        lang_path = os.path.join(os.path.dirname(__file__), 'languages.json')
        with open(lang_path) as f:
            self.languages = json.load(f)
        
        if use_legacy:
            # Import legacy components only if needed
            from LangPred import Predictor
            self._legacy_predictor = Predictor()
        else:
            self._legacy_predictor = None
            self._init_new_model()
    
    def _init_new_model(self):
        """Initialize the new lightweight model"""
        # Optimized feature hashing for memory efficiency
        self.feature_extractor = HashingVectorizer(
            ngram_range=(1, 3),  # Increased to capture more context
            n_features=2**14,    # Increased for better feature space
            alternate_sign=False,
            analyzer='char_wb',
            lowercase=True
        )
        
        # Memory-efficient model with good performance
        self.model = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        
        # Memory-efficient pipeline with balanced sampling
        self.pipeline = Pipeline([
            ('feature_extractor', self.feature_extractor),
            ('sampler', RandomUnderSampler(
                sampling_strategy='auto',
                random_state=42,
                replacement=True  # Allow replacement for very small classes
            )),
            ('classifier', self.model)
        ])
    
    def _read_file_with_encoding(self, file_path: str) -> Optional[str]:
        """Read file content with automatic encoding detection"""
        try:
            # First try UTF-8
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # If UTF-8 fails, detect encoding
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    # Try common encodings first
                    for encoding in ['utf-8', 'latin1', 'cp1252', 'ascii']:
                        try:
                            return raw_data.decode(encoding)
                        except UnicodeDecodeError:
                            continue
                    
                    # If common encodings fail, use chardet
                    result = chardet.detect(raw_data)
                    if result['encoding'] and result['confidence'] > 0.7:
                        return raw_data.decode(result['encoding'])
                    
                print(f"Warning: Could not reliably detect encoding for {file_path}. Skipping.")
                return None
            except Exception as e:
                print(f"Warning: Error reading {file_path}: {str(e)}. Skipping.")
                return None
    
    def _load_training_data(self, data_dir: str = 'FileTypeData') -> Tuple[List[str], List[str]]:
        """Load training data from the train directory with smart resampling."""
        data_dir = Path(data_dir) / 'train'
        if not data_dir.exists():
            raise FileNotFoundError(f"Training data directory not found: {data_dir}")
        
        # Resampling caps
        majority_cap = 500
        minority_target = 200
        
        texts = []
        labels = []
        lang_keys = {k.lower(): k for k in self.languages.keys()}
        for lang_dir in data_dir.iterdir():
            if not lang_dir.is_dir():
                continue
            lang = lang_dir.name.lower()
            if lang not in lang_keys:
                continue
            lang_key = lang_keys[lang]
            files = list(lang_dir.glob('*'))
            n_files = len(files)
            # Smart resampling
            if n_files >= majority_cap:
                selected = np.random.choice(files, majority_cap, replace=False)
            elif n_files < minority_target:
                # Oversample by random duplication
                selected = np.random.choice(files, minority_target, replace=True)
            else:
                selected = files
            for file in selected:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    texts.append(content)
                    labels.append(lang_key)
                except UnicodeDecodeError:
                    continue
        if not texts:
            raise ValueError("No training data found")
        return texts, labels

    def _extract_lightweight_features(self, text: str) -> Dict[str, float]:
        """Extract lightweight features for language detection (enhanced)."""
        features = {}
        features['length'] = len(text)
        features['line_count'] = text.count('\n')
        features['avg_line_length'] = features['length'] / (features['line_count'] + 1)
        chars = Counter(text)
        total_chars = sum(chars.values())
        for char in ['{', '}', ';', ':', '=', '(', ')', '[', ']', '<', '>', '#', '@', '$', '%', '&', '*', '+', '-', '/', '\\', '|', '^', '~', '`', '.', ',']:
            features[f'char_{char}'] = chars[char] / total_chars if total_chars > 0 else 0
        # Enhanced language-specific patterns
        patterns = {
            'python': ['def ', 'import ', 'from ', 'class ', 'try:', 'except:', 'if __name__', 'lambda ', 'yield ', 'async ', 'await ', 'with ', 'as ', '#'],
            'java': ['public class', 'private ', 'public ', 'import ', 'System.out', '@Override', 'interface ', 'implements ', 'extends ', 'throws ', 'new ', 'this.', '//'],
            'cpp': ['#include', 'using namespace', 'std::', 'cout <<', 'cin >>', 'template<', '->', '::', 'new ', 'delete ', 'virtual ', 'override', '//'],
            'javascript': ['function(', 'const ', 'let ', 'var ', 'console.log', 'export ', '=>', 'async ', 'await ', 'import ', 'from ', 'this.', '//'],
            'xml': ['<?xml', '<!DOCTYPE', '<![CDATA[', 'xmlns:', 'xsi:', '<xs:', '<xsd:', '<xsl:', '<xslt:', '<', '</', '/>'],
            'json': ['{"', '":', '": ', '":null', '":true', '":false', '":{', '":[', '":""', '":0', '{', '}', '[', ']'],
            'yaml': ['---', '...', '- ', ':', '|', '>', '!!', '&', '*', '<<:', '!!map', '!!seq', '#'],
            'groovy': ['def ', 'each {', 'findAll {', 'collect {', '@', '->', '?.', '?:', 'as ', 'in ', 'it.', '//']
        }
        for lang, pats in patterns.items():
            count = sum(text.count(pat) for pat in pats)
            features[f'pattern_{lang}'] = count / (features['line_count'] + 1)
        # Language-specific structure
        features['has_curly_braces'] = int('{' in text and '}' in text)
        features['has_square_brackets'] = int('[' in text and ']' in text)
        features['has_angle_brackets'] = int('<' in text and '>' in text)
        features['has_semicolons'] = int(';' in text)
        features['has_colons'] = int(':' in text)
        # Comment style features
        features['has_hash_comment'] = int('#' in text)
        features['has_double_slash_comment'] = int('//' in text)
        features['has_xml_comment'] = int('<!--' in text and '-->' in text)
        return features

    def validate(self, data_dir: str = 'FileTypeData') -> Dict[str, float]:
        """Validate the model on the validation set."""
        data_dir = Path(data_dir) / 'validation'
        if not data_dir.exists():
            raise FileNotFoundError(f"Validation data directory not found: {data_dir}")
        
        texts = []
        true_labels = []
        
        # Load validation files
        for lang_dir in data_dir.iterdir():
            if not lang_dir.is_dir():
                continue
            
            lang = lang_dir.name.lower()
            lang_keys = {k.lower(): k for k in self.languages.keys()}
            if lang not in lang_keys:
                continue
            lang_key = lang_keys[lang]
            for filepath in lang_dir.glob('*'):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    texts.append(content)
                    true_labels.append(lang_key)
                except UnicodeDecodeError:
                    continue
        
        # Make predictions
        pred_labels = []
        for text in texts:
            pred_tuple = self.predict(text, top_k=1)[0]
            pred_labels.append(pred_tuple[0])
        
        # Debug output
        print(f"[DEBUG] Validation set: {len(texts)} samples")
        print(f"[DEBUG] Unique true labels: {set(true_labels)}")
        print(f"[DEBUG] Unique predicted labels: {set(pred_labels)}")
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'report': report
        }

    def test(self, data_dir: str = 'FileTypeData') -> Dict[str, float]:
        """Test the model on the test set."""
        data_dir = Path(data_dir) / 'test'
        if not data_dir.exists():
            raise FileNotFoundError(f"Test data directory not found: {data_dir}")
        
        texts = []
        true_labels = []
        
        # Load test files
        for lang_dir in data_dir.iterdir():
            if not lang_dir.is_dir():
                continue
            
            lang = lang_dir.name.lower()
            lang_keys = {k.lower(): k for k in self.languages.keys()}
            if lang not in lang_keys:
                continue
            lang_key = lang_keys[lang]
            for filepath in lang_dir.glob('*'):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    texts.append(content)
                    true_labels.append(lang_key)
                except UnicodeDecodeError:
                    continue
        
        # Make predictions
        pred_labels = []
        confidences = []
        for text in texts:
            pred_tuple = self.predict(text, top_k=1)[0]
            pred_labels.append(pred_tuple[0])
            confidences.append(pred_tuple[1])
        
        # Debug output
        print(f"[DEBUG] Test set: {len(texts)} samples")
        print(f"[DEBUG] Unique true labels: {set(true_labels)}")
        print(f"[DEBUG] Unique predicted labels: {set(pred_labels)}")
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confidences': confidences
        }
    
    def train(self, input_dir: str, test_size: float = 0.2) -> Dict:
        """
        Train the lightweight model
        
        Args:
            input_dir: Directory containing training files
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing training metrics
        """
        if self.use_legacy:
            accuracy = self._legacy_predictor.learn(input_dir)
            return {'accuracy': accuracy}
        
        print("\nLoading training data...")
        texts, labels = self._load_training_data(input_dir)
        
        if not texts:
            raise ValueError("No training data found")
        
        print("\nComputing class weights...")
        unique_labels = np.unique(labels)
        class_counts = {label: labels.count(label) for label in unique_labels}
        
        # Calculate balanced class weights with smoothing
        max_count = max(class_counts.values())
        class_weights = {label: max_count/(count + 1) for label, count in class_counts.items()}
        self.model.class_weight = class_weights
        
        print("\nSplitting data into train/test sets...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, 
            test_size=test_size, 
            stratify=labels, 
            random_state=42
        )
        
        print("\nTraining model...")
        start_time = time.time()
        self.pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Debug: print model classes
        print(f"[DEBUG] Model classes: {self.pipeline.classes_}")

        # Evaluate model
        y_pred = self.pipeline.predict(X_test)
        metrics = {
            'training_time': training_time,
            'test_size': len(X_test),
            'train_size': len(X_train),
            'class_distribution': class_counts,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        # Debug: print sample predictions
        print("[DEBUG] Sample predictions:")
        for i in range(min(5, len(X_test))):
            pred = self.predict(X_test[i], top_k=3)
            print(f"True: {y_test[i]}, Predicted: {pred}")

        print("\nSaving model...")
        self.save()
        return metrics
    
    def predict(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Predict the language of a given text
        Returns a list of (label, confidence) tuples, even for top_k=1.
        """
        if self.use_legacy:
            # Legacy predictor returns just the label
            return [(self._legacy_predictor.language(text), 1.0)]
        
        # Batch processing for multiple predictions
        if isinstance(text, list):
            probs = self.pipeline.predict_proba(text)
            predictions = []
            for prob in probs:
                pred = sorted(zip(self.pipeline.classes_, prob),
                              key=lambda x: x[1],
                              reverse=True)[:top_k]
                total = sum(p for _, p in pred)
                pred = [(lang, p/total if total > 0 else 0.0) for lang, p in pred]
                predictions.append(pred)
            return predictions
        
        # Single prediction
        prob = self.pipeline.predict_proba([text])[0]
        pred = sorted(zip(self.pipeline.classes_, prob),
                      key=lambda x: x[1],
                      reverse=True)[:top_k]
        total = sum(p for _, p in pred)
        pred = [(lang, p/total if total > 0 else 0.0) for lang, p in pred]
        return pred
    
    def save(self):
        """Save the trained model"""
        if self.use_legacy:
            # Legacy model saves automatically
            return
        
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.pipeline, os.path.join(self.model_dir, 'model.joblib'))
        joblib.dump(self.languages, os.path.join(self.model_dir, 'languages.joblib'))
    
    @classmethod
    def load(cls, model_dir: str = 'models', use_legacy: bool = False) -> 'LanguagePredictor':
        """
        Load a trained model
        
        Args:
            model_dir: Directory containing the model
            use_legacy: Whether to load the legacy model
            
        Returns:
            Loaded LanguagePredictor instance
        """
        predictor = cls(model_dir, use_legacy)
        
        if not use_legacy:
            try:
                predictor.pipeline = joblib.load(os.path.join(model_dir, 'model.joblib'))
                predictor.languages = joblib.load(os.path.join(model_dir, 'languages.joblib'))
            except FileNotFoundError:
                raise ValueError(f"No model found in {model_dir}")
        
        return predictor 