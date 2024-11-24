import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd

def get_processed_data(file_path):
    data = pd.read_csv(file_path, sep=';')
    return data

class JokesDataset(Dataset):
    def __init__(self, jokes, topics):
        self.jokes = jokes
        self.topics = topics

    def __len__(self):
        return len(self.jokes)

    def __getitem__(self, idx):
        return self.jokes[idx], self.topics[idx]

# Load your processed data
processed_data = get_processed_data('/data/processed/all_processed.cv')
jokes = processed_data['jokes']
topics = processed_data['topics']

# Create dataset and dataloader
dataset = JokesDataset(jokes, topics)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define your model
class JokeModel(nn.Module):
    def __init__(self):
        super(JokeModel, self).__init__()
        # Define layers

    def forward(self, x):
        # Define forward pass
        return x

model = JokeModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    for batch_jokes, batch_topics in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_jokes)
        loss = criterion(outputs, batch_topics)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'joke_model.pth')