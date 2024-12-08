import torch 
import torch.nn as nn 
import torch.optim as optim 
from transformers import AutoModelForCausalLM, AutoTokenizer

class ModelDeployer: 
    def __init__(self, model, device): 
        self.model = model  
        self.device = device

    def deploy(self):
        self.model.to(self.device)
        print("Model deployed successfully.")

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deployer = ModelDeployer(model, device)
    deployer.deploy()