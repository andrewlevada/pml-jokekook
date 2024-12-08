import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer
)

# File paths
PROCESSED_FILE = "../data/processed/processed_7.csv"  # Use the processed file directly
OUTPUT_DIR = "../outputs/fine_tuned_model"
LOG_DIR = "../logs"

# Load dataset
print("Loading dataset...")
dataset = load_dataset(
    "csv",
    data_files={"train": PROCESSED_FILE},
    delimiter=";"  # Use semicolon separator
)
print(f"Dataset loaded: {dataset}")

# Load pre-trained model and tokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Replace with the actual model path if local
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # Adjust num_labels if needed

# Custom joke prompt
joke_prompt = """write a joke in a given topic.

### Joke:
{}

### Joke Topic:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Ensure proper termination of the model's output

# Format the dataset using the custom prompt
def formatting_prompts_func(examples):
    jokes = examples["Joke"]
    topics = examples["Joke topic"]
    texts = []
    for joke, topic in zip(jokes, topics):
        # Add EOS_TOKEN for proper generation
        text = joke_prompt.format(joke, topic) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

print("Formatting dataset...")
formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

# Tokenize the formatted dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)

print("Tokenizing dataset...")
tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

# Data collator for batching
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="no",  # Disable validation since no separate split
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir=LOG_DIR,
    save_total_limit=2,
    push_to_hub=False,
    remove_unused_columns=False  # Prevents column removal during tokenization
)

# Tokenize the formatted dataset
formatted_dataset = formatted_dataset.map(tokenize_function, batched=True)

# Update the data_collator to use the correct input format
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,  
    train_dataset=formatted_dataset
    # tokenizer=tokenizer  # Comment out this line
)

# Training
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
