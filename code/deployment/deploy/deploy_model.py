import torch 
import torch.nn as nn 
import torch.optim as optim 

class ModelDeployer: 
    def __init__(self, model, device): 
        self.model = model  
        self.device = device

    def deploy(self):
        self.model.to(self.device)
        #

if __name__ == "__main__":
    model = nn.Linear(10, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deployer = ModelDeployer(model, device)
    deployer.deploy()