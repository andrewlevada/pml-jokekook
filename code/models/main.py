model_name = "Qwen/Qwen2.5-3B"

# Some ideas for future implementation:

# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# training_args = TrainingArguments(
#     output_dir="../../models/qwen2.5-3b-finetuned",
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=4,
#     learning_rate=2e-5,
#     warmup_steps=100,
#     logging_steps=10,
#     save_steps=100,
#     fp16=True
# )
