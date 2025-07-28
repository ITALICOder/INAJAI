import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Define custom dataset for wine reviews
class WineReviewDataset(Dataset):
    def __init__(self, reviews, points, additional_features):
        self.reviews = reviews
        self.points = points
        self.additional_features = additional_features
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')

        # Add a padding token if it does not exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        point = self.points[idx]

        # Flatten additional features
        additional_info = self.additional_features[idx].toarray().flatten()
        additional_info_str = ' '.join(map(str, additional_info))
        input_text = f"{review} {additional_info_str}"

        # Tokenization with truncation
        encoding = self.tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024, padding='max_length')

        # The inputs will be used as labels for language modeling
        labels = encoding['input_ids'].clone()

        if point is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss calculation
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
        }

# Load Data Function
def load_data(directory):
    all_data = []
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, file))
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Data Preparation
def prepare_data(data):
    required_cols = ['Review_content', 'points', 'country', 'Review_name', 
                     'Year', 'winery', 'region_1', 'region_2', 'province']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {', '.join(missing_cols)}")

    data['points'] = pd.to_numeric(data['points'], errors='coerce')
    data.dropna(subset=['points'], inplace=True)

    # Prepare additional features
    additional_features = data[['country', 'Review_name', 'Year', 'winery', 'region_1', 'region_2', 'province']].fillna('Unknown')
    additional_features['Year'] = additional_features['Year'].astype(str)

    encoder = OneHotEncoder(sparse_output=True)
    X_additional = encoder.fit_transform(additional_features)

    return data['Review_content'], data['points'], X_additional

def fine_tune_model(model, tokenizer, X_train, y_train, X_val, y_val, X_train_additional, X_val_additional):
    train_dataset = WineReviewDataset(X_train.tolist(), y_train.tolist(), X_train_additional)
    val_dataset = WineReviewDataset(X_val.tolist(), y_val.tolist(), X_val_additional)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        save_steps=10_000,
        save_total_limit=2,
        eval_strategy="steps",  # Update to use eval_strategy instead of evaluation_strategy
        logging_dir='./logs',
        logging_steps=200,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

# Generate review function
def generate_review(tokenizer, model, input_text, additional_info):
    combined_input = f"{input_text} {additional_info}"
    input_ids = tokenizer.encode(combined_input, return_tensors='pt')

    # Ensure input does not exceed the maximum length
    input_ids = input_ids[:, :1024]  # Truncate to the maximum length

    with torch.no_grad():
        output = model.generate(input_ids, max_length=250, num_return_sequences=1)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Function to save the model
def save_model(model, tokenizer, save_path='./my_finetuned_model'):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    data = load_data('csv2')
    
    X, y, X_additional = prepare_data(data)
    
    # Split the data
    X_train_text, X_val_text, y_train, y_val, X_train_additional, X_val_additional = train_test_split(
        X, y, X_additional, test_size=0.2, random_state=42)

    # Load the Transformer Model / using gpt2 for example now
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    fine_tune_model(model, tokenizer, X_train_text, y_train, X_val_text, y_val, X_train_additional, X_val_additional)

    sample_input = "Elegant and balanced with notes of dark fruit"
    additional_info = "Country: Italy, Year: 2015"
    print(generate_review(tokenizer, model, sample_input, additional_info))

    # Save the model
    save_model(model, tokenizer)
