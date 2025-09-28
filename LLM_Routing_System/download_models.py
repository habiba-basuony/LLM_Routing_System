from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Create models folder if it doesn't exist
os.makedirs('./models', exist_ok=True)

# List of required models
models = ['distilgpt2', 'gpt2-medium', 'gpt2-large']

for model_name in models:
    print(f"Downloading {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f'./models/{model_name}')
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=f'./models/{model_name}')
        print(f"Successfully downloaded {model_name}!")
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")

print("Model download completed!")