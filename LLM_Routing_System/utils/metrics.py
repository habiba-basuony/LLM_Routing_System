def calculate_accuracy(model_name, query, response):
    base_accuracy = 0.7
    
    if 'medium' in model_name or 'gemma' in model_name:
        base_accuracy = 0.8
    elif 'large' in model_name:
        base_accuracy = 0.85
    
    # Higher accuracy for Ollama models
    if 'ollama' in model_name:
        base_accuracy += 0.1
    
    # Adjust accuracy based on response quality
    query_length = len(query.split())
    response_length = len(response.split())
    
    if response_length == 0:
        return 0.0  # Avoid division by zero
    
    response_quality = min(1.0, response_length / max(1, query_length))
    
    return min(0.95, base_accuracy * response_quality)

def calculate_cost(model_name, response_length, provider):
    """Calculate response cost based on model and response length"""
    # Cost per token approximation
    cost_per_token = {
        'ollama': {
            'phi': 0.0000001,  # $0.0001 per 1000 tokens
            'gemma:2b': 0.0000003  # $0.0003 per 1000 tokens
        },
        'huggingface': {
            'gpt2-large': 0.00001  # $0.01 per 1000 tokens
        }
    }
    
    # Estimate tokens (word ≈ 1.3 tokens)
    estimated_tokens = response_length * 1.3
    
    if provider == 'ollama':
        return cost_per_token['ollama'].get(model_name, 0.0000002) * estimated_tokens
    else:
        return cost_per_token['huggingface'].get(model_name, 0.00001) * estimated_tokens