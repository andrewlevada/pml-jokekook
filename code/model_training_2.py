import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer
)
from transformers import DataCollatorWithPadding  # Add this import

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
    delimiter=";"  # Use semicolon separator
)
print(f"Datasets loaded: {dataset}")

# Load pre-trained model and tokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Replace with the actual model path if local
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # Adjust num_labels if needed

joke_prompt = """write a joke in a given topic.

### Joke:
{}

### Joke Topic:
{}"""

EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    jokes = examples["Joke"]
    topics = examples["Joke topic"]
    texts = []
    for joke, topic in zip(jokes, topics):
        # Add EOS_TOKEN for proper generation
        text = joke_prompt.format(joke, topic) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Apply the formatting function to the datasets
print("Formatting datasets...")
formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

# Update the dataset to include input_ids for the tokenizer
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the formatted dataset
formatted_dataset = formatted_dataset.map(tokenize_function, batched=True)

# Update the data_collator to use the correct input format
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# Trainer arguments
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
    save_total_limit=2, 
    push_to_hub=False,
    remove_unused_columns=False  # Add this line
)

# Trainer setup
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,  
    train_dataset=formatted_dataset["train"],  # Use formatted_dataset instead
    eval_dataset=formatted_dataset["validation"],  # Use formatted_dataset instead
    # tokenizer=tokenizer  # Comment out this line
    # Add processing_class if needed based on your model
)

# Training
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")