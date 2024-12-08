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
PROCESSED_FILE = "../data/processed/processed_7.csv"  # Make sure this path is correct
OUTPUT_DIR = "../outputs/fine_tuned_model"
LOG_DIR = "../logs"

# Load dataset
print("Loading dataset...")
dataset = load_dataset(
    "csv",
    data_files={"train": PROCESSED_FILE},
    delimiter=";"  # Ensure the delimiter is correct
)
print(f"Dataset loaded: {dataset}")

# Load pre-trained model and tokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Ensure this model exists or use another available model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # Adjust num_labels if needed

# Custom joke prompt for formatting dataset
joke_prompt = """write a joke in a given topic.

### Joke:
{}

### Joke Topic:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Ensure proper termination of the model's output

# Function to format dataset for model input
def formatting_prompts_func(examples):
    jokes = examples["Joke"]
    topics = examples["Joke topic"]
    texts = []
    for joke, topic in zip(jokes, topics):
        text = joke_prompt.format(joke, topic) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

print("Formatting dataset...")
formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

# Tokenization function (Ensure the output is correctly structured)
def tokenize_function(examples):
    # Tokenize the examples['text'] field
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=512)

print("Tokenizing dataset...")
tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)

# Check tokenization output
print("Sample tokenized data:")
print(tokenized_dataset["train"][0])  # Print a sample to verify the tokenization

# Data collator for batching and padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="no",  # No separate validation split in this case
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

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"]
)

# Training the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
