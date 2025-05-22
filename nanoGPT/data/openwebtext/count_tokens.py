import json
import tiktoken
from tqdm import tqdm
from datasets import load_dataset

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")

def analyze_dataset(dataset_path):
    # Load the dataset
    dataset = load_dataset(dataset_path, num_proc=8)
    
    # Initialize counters
    total_tokens = 0
    total_examples = 0
    min_tokens = float('inf')
    max_tokens = 0
    
    # Process each example in the training set
    for example in tqdm(dataset['train'], desc="Processing examples"):
        tokens = enc.encode_ordinary(example['text'])
        num_tokens = len(tokens)
        
        # Update statistics
        total_tokens += num_tokens
        total_examples += 1
        min_tokens = min(min_tokens, num_tokens)
        max_tokens = max(max_tokens, num_tokens)
        
        # Print progress every 100000 examples
        if total_examples % 100000 == 0:
            print(f"Processed {total_examples} examples...")
    
    # Print final statistics
    print("\nDataset Statistics:")
    print(f"Total number of tokens: {total_tokens:,}")
    print(f"Total number of examples: {total_examples:,}")
    print(f"Average tokens per example: {total_tokens/total_examples:.2f}")
    print(f"Min tokens in an example: {min_tokens}")
    print(f"Max tokens in an example: {max_tokens}")

if __name__ == "__main__":
    analyze_dataset("openwebtext")