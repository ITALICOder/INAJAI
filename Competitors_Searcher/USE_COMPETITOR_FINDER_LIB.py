import pandas as pd
import numpy as np
import joblib  # for saving the model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(file_path):
    """Load training data from CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess text data and prepare it for modeling."""
    # Assume 'content' is the text and 'website' is the identifier
    return df['content'], df['website']

def find_competitors(model, vectorizer, sample_text, input_website, df, n=10):
    """Find top N competitors based on the similarity of content."""
    # Transform the sample text
    sample_tfidf = vectorizer.transform([sample_text])
    
    # Compute similarity between sample text and all other texts
    similarities = cosine_similarity(sample_tfidf, model).flatten()
    
    # Get indices of top N most similar competitors, excluding the input website
    top_indices = similarities.argsort()[-n-1:][::-1]  # Get indices of n+1 most similar competitors to exclude the input
    
    # Filter out the input website and prepare results
    results = []
    count = 0
    for i in top_indices:
        if df.iloc[i]['website'] != input_website:
            results.append((df.iloc[i]['website'], similarities[i]))
            count += 1
        if count >= n:  # Stop once we've found n competitors
            break
    
    return results

def save_vectorizer(vectorizer, filename):
    """Save the TF-IDF vectorizer to a file."""
    joblib.dump(vectorizer, filename)
    print(f"Vectorizer saved to {filename}")

if __name__ == '__main__':
    # Load training data
    training_data_path = 'training_data/training_data.csv'
    df = load_data(training_data_path)

    # Preprocess the data
    X, websites = preprocess_data(df)

    # Create a TF-IDF vectorizer with additional parameters for better modeling
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_df=0.85, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(X)

    # Save the vectorizer
    save_vectorizer(vectorizer, 'tfidf_vectorizer.joblib')

    # Test the model with a sample input to find competitors
    sample_input = "buy a new domain"  # The content you want to find similar competitors for
    competitors = find_competitors(tfidf_matrix, vectorizer, sample_input, sample_input, df, n=10)

    # Print out competitors and their similarity scores
    print("Top competitors based on content similarity:")
    for website, score in competitors:
        print(f"Website: {website}, Similarity Score: {score:.4f}")
