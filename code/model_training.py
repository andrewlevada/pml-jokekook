import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments
)
from trl import SFTTrainer

if __name__ != "__main__": 
    exit()

# File paths
PROCESSED_FILE = "../data/processed/processed_7.csv"  # Make sure this path is correct
OUTPUT_DIR = "../outputs/fine_tuned_model"
LOG_DIR = "../logs"

# Load dataset
print("Loading dataset...")
dataset = datasets.load_dataset("csv", data_files={"train": PROCESSED_FILE}, delimiter=";", split="train")
print(f"Dataset loaded: {dataset}")

# Load pre-trained model and tokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Ensure this model exists or use another available model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, max_length=1024, max_seq_length=1024)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2, max_length=1024)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id


# Custom joke prompt for formatting dataset
joke_prompt = """write a joke in a given topic.

### Joke:
{}

### Joke Topic:
{}"""

EOS_TOKEN = tokenizer.eos_token  # Ensure proper termination of the model's output

# Function to format dataset for model input
def formatting_prompts_func(examples):
    j = examples["Joke"]
    t = examples["Joke topic"]
    
    return {"text": joke_prompt.format(j, t) + EOS_TOKEN}

print("Formatting dataset...")
formatted_dataset = dataset.map(formatting_prompts_func, batched=False)
formatted_dataset = formatted_dataset.map(lambda x: {"text": x["text"][:1024]}, batched=False)

max_seq_length = 1024

# Training arguments
print("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir=LOG_DIR,
    save_total_limit=2,
)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    dataset_text_field = "text",
    max_seq_length=max_seq_length,
)

# Training the model
print("Starting training...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")