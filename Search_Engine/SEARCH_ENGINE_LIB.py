from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
import faiss  # Replacing hnswlib for optimized memory and search performance
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv('training_data.csv')

# Use a smaller, efficient model for embeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Function to get embeddings efficiently
def get_embeddings(sentences, tokenizer, model):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Generate embeddings in batches to save memory
batch_size = 32
all_embeddings = []
for i in range(0, len(data), batch_size):
    batch = data['content'][i:i+batch_size].tolist()
    batch_embeddings = get_embeddings(batch, tokenizer, model)
    all_embeddings.append(batch_embeddings)

# Concatenate embeddings and convert to float32
embeddings = np.vstack(all_embeddings).astype(np.float32)

# Use faiss for efficient indexing and querying
d = embeddings.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatIP(d)  # Inner product to approximate cosine similarity
index.add(embeddings)  # Add embeddings to index

# Query function with improved accuracy
def search_engine_query(query, data, tokenizer, model, index, top_k=10):
    # Generate embedding for the query
    query_embedding = get_embeddings([query], tokenizer, model).astype(np.float32)
    
    # Perform search with faiss
    _, I = index.search(query_embedding, top_k)
    
    # Retrieve and format results
    results = []
    for idx in I[0]:
        if idx < len(data):
            results.append({
                'website': data.iloc[idx]['website'],
                'description': data.iloc[idx]['content'][:200]
            })
    
    return results

# Example usage
query = "Buy a domain"
results = search_engine_query(query, data, tokenizer, model, index)
for res in results:
    print(f"Website: {res['website']}\nDescription: {res['description']}\n")
