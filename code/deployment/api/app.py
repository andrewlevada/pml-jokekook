from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B"

# Loading model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

app = FastAPI(
    title="PML: JokeKook",
)

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(model.device)
    
    # Simple text generation (prompt engineering on app side)
    output = model.generate(
        input_ids,
        max_new_tokens=request.max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": generated_text}
