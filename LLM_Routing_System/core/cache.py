import hashlib
from typing import Any, Dict, Optional
import diskcache

class QueryCache:
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.cache = diskcache.Cache(
            directory='./cache_data',
            size_limit=max_size * 1024 * 1024,
            eviction_policy='least-recently-used'
        )
    
    def _get_query_hash(self, query: str) -> str:
        return hashlib.md5(query.lower().encode('utf-8')).hexdigest()
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        query_hash = self._get_query_hash(query)
        return self.cache.get(query_hash)
    
    def set(self, query: str, result: Dict[str, Any]) -> None:
        query_hash = self._get_query_hash(query)
        clean_result = {
            'query': result.get('query'),
            'model_used': result.get('model_used'),
            'response': result.get('response'),
            'speed_ms': result.get('speed_ms'),
            'accuracy': result.get('accuracy'),
            'cost': result.get('cost'),
            'classification': result.get('classification'),
            'confidence': result.get('confidence')
        }
        self.cache[query_hash] = clean_result
    
    def clear(self) -> None:
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        try:
            stats = self.cache.stats()
            if isinstance(stats, tuple) and len(stats) >= 2:
                hits, misses = stats[0], stats[1]
                total = hits + misses
                hit_rate = hits / total if total > 0 else 0
            else:
                hit_rate = 0
            
            return {
                'size': self.cache.volume(),
                'items': len(self.cache),
                'hit_rate': hit_rate
            }
        except Exception as e:
            return {
                'size': self.cache.volume(),
                'items': len(self.cache),
                'hit_rate': 0
            }