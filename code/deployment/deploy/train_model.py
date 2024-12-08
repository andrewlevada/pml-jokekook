import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

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
model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)
        
        # Simple text generation (prompt engineering on app side)
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'joke_model.pth')