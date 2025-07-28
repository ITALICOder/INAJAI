import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib

# Load the dataset
data = pd.read_csv('reviews.csv')  # Ensure the CSV has 'review', 'score', 'company'

# Preprocessing
data = data[data['review'].notna() & data['score'].notna()]  # Remove NaN reviews and scores
data['score'] = data['score'].astype(float)

# Combine reviews for each company
grouped_data = data.groupby('company').agg({
    'review': ' '.join,  # Join all reviews into one string per company
    'score': 'mean'      # Average score for the company
}).reset_index()

# Normalize scores to [0.00, 10.00]
scaler = MinMaxScaler((0, 10))
grouped_data[['score']] = scaler.fit_transform(grouped_data[['score']])

# Prepare data for LSTM
X = grouped_data['review']
y = grouped_data['score']

# Tokenization
max_words = 5000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(sequences, padding='post')

# Splitting the data
X_train, X_val, y_train, y_val = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=X_pad.shape[1]))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))  # Output layer for regression

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

# Fit the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the trained model to a file
model.save('review_generation_model.h5')

# Save the tokenizer and scaler for future use
joblib.dump(tokenizer, 'tokenizer.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Function to generate review and score for the company
def generate_review_and_score(company_name):
    # Filter the reviews for the specified company
    filtered_reviews = data[data['company'].str.lower() == company_name.lower()]
    
    if filtered_reviews.empty:
        return "No reviews found for {}".format(company_name), 0.0

    # Combine reviews for the company
    combined_review = ' '.join(filtered_reviews['review'].tolist())
    
    # Tokenize and pad the input review
    review_seq = tokenizer.texts_to_sequences([combined_review])
    review_pad = pad_sequences(review_seq, maxlen=X_pad.shape[1], padding='post')
    
    # Predict the score
    predicted_score = model.predict(review_pad)
    predicted_score = scaler.inverse_transform(predicted_score.reshape(-1, 1))

    return combined_review, round(predicted_score[0][0], 2)

# Example usage
# Load the model, tokenizer, and scaler for future use
model = load_model('review_generation_model.h5')
tokenizer = joblib.load('tokenizer.pkl')
scaler = joblib.load('scaler.pkl')

company_name = 'example.com' # create loop for evaluate the result manually
review_example, score_example = generate_review_and_score(company_name)

print(f"Combined Review: {review_example}")
print(f"Predicted Score: {score_example}/10")

