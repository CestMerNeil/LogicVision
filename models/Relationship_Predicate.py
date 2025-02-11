import torch
import torch.nn as nn
import ltn

class In(ltn.Predicate):
    def __init__(self, input_dim):
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        super().__init__(model=net)
        self.net = net
    
    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        subj_tensor = subj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.expand_as(subj_tensor)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1).view(-1, 10)
        return self.net(combined)

class On(ltn.Predicate):
    def __init__(self, input_dim):
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        super().__init__(model=net)
        self.net = net
    
    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        subj_tensor = subj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.expand_as(subj_tensor)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1).view(-1, 10)
        return self.net(combined)

class NextTo(ltn.Predicate):
    def __init__(self, input_dim):
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        super().__init__(model=net)
        self.net = net

    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        subj_tensor = subj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.expand_as(subj_tensor)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1).view(-1, 10)
        return self.net(combined)

class OnTopOf(ltn.Predicate):
    def __init__(self, input_dim):
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        super().__init__(model=net)
        self.net = net
    
    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        subj_tensor = subj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.expand_as(subj_tensor)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1).view(-1, 10)
        return self.net(combined)
        
class Near(ltn.Predicate):
    def __init__(self, input_dim):
        # Define the neural network model for the Near predicate.
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        super().__init__(model=net)
        self.net = net
            
    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        # Get the device from the network parameters.
        device = next(self.net.parameters()).device

        # Move the input tensors to the appropriate device.
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        
        # Adjust tensor dimensions to match expected input shape.
        subj_tensor = subj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.unsqueeze(1)
        
        # Expand the object tensor to match the subject tensor's shape.
        obj_tensor = obj_tensor.expand_as(subj_tensor)
        
        # Concatenate subject and object tensors and reshape for network input.
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1).view(-1, 10)
        
        # Pass the combined tensor through the network and return the output.
        return self.net(combined)

class Under(ltn.Predicate):
    def __init__(self, input_dim):
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        super().__init__(model=net)
        self.net = net

    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        subj_tensor = subj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.unsqueeze(1)
        obj_tensor = obj_tensor.expand_as(subj_tensor)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1).view(-1, 10)
        return self.net(combined)