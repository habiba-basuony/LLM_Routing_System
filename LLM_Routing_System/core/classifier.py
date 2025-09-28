from typing import Tuple, Dict, List, Any
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

class QueryClassifier:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        self.is_trained = False
    
    def extract_features(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        
        features = {
            'length': len(query),
            'word_count': len(query.split()),
            'has_question_mark': '?' in query,
            'has_exclamation': '!' in query,
        }
        
        for keyword in self.config.get('simple_keywords', []):
            features[f'has_{keyword}'] = int(keyword in query_lower)
        
        for keyword in self.config.get('advanced_keywords', []):
            features[f'has_{keyword}'] = int(keyword in query_lower)
        
        return features
    
    def train(self, training_data: List[Tuple[str, str]]):
        if not training_data:
            return
        
        queries, labels = zip(*training_data)
        X = self.vectorizer.fit_transform(queries)
        self.model.fit(X, labels)
        self.is_trained = True
    
    def classify(self, query: str) -> Tuple[str, float]:
        # Use trained model if available
        if self.is_trained:
            X = self.vectorizer.transform([query])
            probabilities = self.model.predict_proba(X)[0]
            predicted_class = self.model.predict(X)[0]
            confidence = max(probabilities)
            return predicted_class, confidence
        
        # Fallback to rule-based classification
        return self._rule_based_classification(query)
    
    def _rule_based_classification(self, query: str) -> Tuple[str, float]:
        """Rule-based classification when no trained model is available"""
        query_lower = query.lower()
        config = self.config
        
        # Check for keywords
        simple_keywords = config.get('simple_keywords', [])
        advanced_keywords = config.get('advanced_keywords', [])
        
        has_simple = any(kw in query_lower for kw in simple_keywords)
        has_advanced = any(kw in query_lower for kw in advanced_keywords)
        
        # Check query length
        length_thresholds = config.get('query_length_thresholds', {})
        is_short = len(query) < length_thresholds.get('simple', 80)
        is_long = len(query) > length_thresholds.get('advanced', 200)
        
        # Classification logic
        if has_advanced or is_long:
            confidence = 0.85 if has_advanced else 0.70
            return "advanced", confidence
        elif has_simple and is_short:
            confidence = 0.90 if has_simple else 0.75
            return "simple", confidence
        else:
            return "medium", 0.80