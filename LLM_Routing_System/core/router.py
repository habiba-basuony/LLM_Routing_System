from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from .cache import QueryCache
from .classifier import QueryClassifier
from .models import ModelManager

@dataclass
class RoutingResult:
    query: str
    model_used: str
    response: str
    speed_ms: float
    accuracy: float
    cost: float
    from_cache: bool
    classification: str
    confidence: float

class Router:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = QueryCache(
            ttl_seconds=config['caching']['ttl_seconds'],
            max_size=config['caching']['max_size']
        )
        self.classifier = QueryClassifier(config['classification'])
        self.model_manager = ModelManager()
        self.metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'model_usage': {},
            'total_cost': 0.0,
            'total_time': 0.0
        }
    
    def route_query(self, query: str, force_model: Optional[str] = None) -> RoutingResult:
        self.metrics['total_queries'] += 1
        
        if force_model:
            if force_model not in self.config['models']:
                raise ValueError(f"Model {force_model} not found in configuration")
                
            response, speed, accuracy = self.model_manager.generate_response(
                self.config['models'][force_model], query
            )
            
            cost = self.config['models'][force_model]['cost']
            
            # Update metrics
            self._update_metrics(force_model, speed, cost)
            
            return RoutingResult(
                query=query,
                model_used=force_model,
                response=response,
                speed_ms=speed,
                accuracy=accuracy,
                cost=cost,
                from_cache=False,
                classification='forced',
                confidence=1.0
            )
        
        # Check cache for normal queries
        cached_result = self.cache.get(query)
        if cached_result:
            self.metrics['cache_hits'] += 1
            return RoutingResult(from_cache=True, **cached_result)
        
        # Classify query
        classification, confidence = self.classifier.classify(query)
        
        # Select model based on classification and confidence
        chosen_model = self._select_model(classification, confidence)
        
        # Generate response
        response, speed, accuracy = self.model_manager.generate_response(
            self.config['models'][chosen_model], query
        )
        
        cost = self.config['models'][chosen_model]['cost']
        
        # Update metrics
        self._update_metrics(chosen_model, speed, cost)
        
        # Cache the result
        result_data = {
            'query': query,
            'model_used': chosen_model,
            'response': response,
            'speed_ms': speed,
            'accuracy': accuracy,
            'cost': cost,
            'classification': classification,
            'confidence': confidence
        }
        
        self.cache.set(query, result_data)
        
        return RoutingResult(from_cache=False, **result_data)
    
    def _select_model(self, classification: str, confidence: float) -> str:
        """Select appropriate model based on classification and confidence"""
        config = self.config['routing']
        thresholds = config['confidence_thresholds']
    
        # Model mapping based on classification
        model_mapping = {
            'simple': 'ollama-simple',
            'medium': 'ollama-medium',
            'advanced': 'gpt2-large'
        }
    
        # Get fallback model from config
        fallback_model = config.get('fallback_model', 'ollama-simple')
    
        try:
            # Check if confidence meets the threshold for this classification
            if classification in thresholds and confidence >= thresholds[classification]:
                return model_mapping.get(classification, fallback_model)
            else:
                # Confidence is too low, use fallback
                return fallback_model
        except Exception as e:
            print(f"Error in model selection: {e}")
            return fallback_model
    
    def _update_metrics(self, model: str, speed: float, cost: float):
        """Update performance metrics"""
        if model not in self.metrics['model_usage']:
            self.metrics['model_usage'][model] = 0
        self.metrics['model_usage'][model] += 1
        self.metrics['total_cost'] += cost
        self.metrics['total_time'] += speed
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        cache_stats = self.cache.get_stats()
        
        return {
            **self.metrics,
            'cache_stats': cache_stats,
            'cache_hit_rate': self.metrics['cache_hits'] / self.metrics['total_queries'] if self.metrics['total_queries'] > 0 else 0,
            'avg_cost_per_query': self.metrics['total_cost'] / (self.metrics['total_queries'] - self.metrics['cache_hits']) if (self.metrics['total_queries'] - self.metrics['cache_hits']) > 0 else 0,
            'avg_time_per_query': self.metrics['total_time'] / (self.metrics['total_queries'] - self.metrics['cache_hits']) if (self.metrics['total_queries'] - self.metrics['cache_hits']) > 0 else 0
        }
    
    def clear_cache(self):
        """Clear cache"""
        self.cache.clear()