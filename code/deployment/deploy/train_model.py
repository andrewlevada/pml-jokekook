import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_processed_data(file_path):
    data = pd.read_csv(file_path, sep=';')
    return data

class JokesDataset(Dataset):
    def __init__(self, jokes, topics, tokenizer):
        self.jokes = jokes
        self.topics = topics
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.jokes)

    def __getitem__(self, idx):
        joke = self.jokes[idx]
        topic = self.topics[idx]
        encoding = self.tokenizer.encode_plus(
            joke,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(topic, dtype=torch.long)
        }

# Load your processed data
processed_data = get_processed_data('/data/processed/all_processed.cv')
jokes = processed_data['jokes']
topics = processed_data['topics']

# Load pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('Qwen/Qwen2.5-0.5B')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')

# Create dataset and dataloader
dataset = JokesDataset(jokes, topics, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['labels'].to(device)
        )
        loss = criterion(outputs.logits, batch['labels'].to(device))
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'joke_model.pth')