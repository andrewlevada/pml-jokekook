import modal
from fastapi import HTTPException
from pydantic import BaseModel

app = modal.App("jokekook")

# Defining Docker container for Modal
@app.cls(
    image=modal.Image.debian_slim().pip_install(
        "fastapi",
        "uvicorn",
        "transformers",
        "torch",
    ),
    gpu=modal.gpu.A100(count=1),
    container_idle_timeout=5 * 60,
    secrets=[modal.Secret.from_name("HUGGINGFACE_API_TOKEN")],
)
class LLMModel:
    @modal.enter()
    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen2.5-3B"  # TODO: Replace!
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
        
        print("Model loaded!")

    @modal.method()
    def generate_text(self, topic: str, max_length: int = 200) -> str:
        prompt = f"""
        You are a joke generator. You are given a topic and you need to generate a joke about it.
        Topic: {topic}
        Joke: 
        """
        
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text
        except Exception as e:
            print(f"Error generating text: {e}")
            raise e


class Prompt(BaseModel):
    topic: str
    max_length: int = 200





@app.function()
@modal.web_endpoint(
    method="POST",
    label="generate",
    docs=True
)
def generate(prompt: Prompt):
    if not prompt.topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    try:
        generated_text = LLMModel.generate_text.remote(prompt.topic, prompt.max_length)
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

