import pandas as pd
from sklearn.model_selection import train_test_split

# File paths
INPUT_FILE = "../data/processed/all_processed.csv"  # Original dataset
TRAIN_FILE = "../data/train.csv"  # Output for training data
VALID_FILE = "../data/valid.csv"  # Output for validation data

# Load the dataset
data = pd.read_csv(INPUT_FILE, sep=";")  # Adjust separator if needed
print("Original data loaded. Example rows:")
print(data.head())

data = data[['Joke', 'Joke topic']]


# Split the dataset into train (90%) and validation (10%) sets
train_data, valid_data = train_test_split(data, test_size=0.1, random_state=42)

# Save the split datasets
train_data.to_csv(TRAIN_FILE, sep=";", index=False)
valid_data.to_csv(VALID_FILE, sep=";", index=False)

print(f"Training data saved to: {TRAIN_FILE}")
print(f"Validation data saved to: {VALID_FILE}")
