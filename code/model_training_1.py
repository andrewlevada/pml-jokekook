import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# File paths
TRAIN_FILE = "../data/train.csv"
VALID_FILE = "../data/valid.csv"
OUTPUT_DIR = "../outputs/fine_tuned_model"
LOG_DIR = "../logs"

# Load dataset
print("Loading datasets...")
dataset = load_dataset(
    "csv",
    data_files={"train": TRAIN_FILE, "validation": VALID_FILE},
    delimiter=";"
)
print(f"Datasets loaded: {dataset}")

# Load pre-trained model and tokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Replace with the actual model path if local
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # Adjust num_labels if needed

# Preprocess data
def preprocess_function(examples):
    return tokenizer(
        text=examples["Joke"],
        text_pair=examples["Joke topic"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

print("Tokenizing datasets...")
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir=LOG_DIR,
    save_total_limit=2,  # Keep only the last 2 checkpoints
    push_to_hub=False  # Set this to True if you're using Hugging Face Hub
)

# Trainer setup
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

# Training
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")