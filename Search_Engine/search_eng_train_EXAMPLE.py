import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# Load and prepare data
data = pd.read_csv('training_data/training_data.csv')  # Assumes columns 'website' and 'content'
data['input_text'] = "Website and summary for: " + data['content'].str[:5000]  # Truncate content to 500 chars
data['output_text'] = data['website'] + " | " + data['content'].str[:150]     # Shortened content for training

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define Dataset
class WebsiteDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]['input_text']
        output_text = self.data.iloc[idx]['output_text']

        inputs = self.tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            output_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels['input_ids'].squeeze(0)  # Ensure labels are a LongTensor
        }

# Initialize Tokenizer and Model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Freeze DistilBERT weights for faster training
for param in distilbert.parameters():
    param.requires_grad = False

# Define the Sequence Model
class SummaryModel(nn.Module):
    def __init__(self, distilbert, hidden_dim=256, vocab_size=30522):
        super(SummaryModel, self).__init__()
        self.distilbert = distilbert
        self.fc = nn.Linear(distilbert.config.hidden_size, hidden_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)  # Predict vocab size tokens

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state  # Full sequence output
        fc_output = torch.relu(self.fc(hidden_state))
        return self.out(fc_output)  # Output shape: (batch_size, seq_len, vocab_size)

# Initialize model, loss, and optimizer
model = SummaryModel(distilbert)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # Ignore padding tokens
optimizer = Adam(model.parameters(), lr=1e-4)

# Create DataLoaders
train_dataset = WebsiteDataset(train_data, tokenizer)
test_dataset = WebsiteDataset(test_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Training Loop
def train_model(model, train_loader, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Reshape outputs and labels for CrossEntropyLoss
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten outputs to (batch_size * seq_len, vocab_size)
            labels = labels.view(-1)  # Flatten labels to (batch_size * seq_len)

            # Calculate loss and backpropagate
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')

# Train and save the model
train_model(model, train_loader)
torch.save(model.state_dict(), 'website_summary_model.pth')
