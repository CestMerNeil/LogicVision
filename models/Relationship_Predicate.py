import torch
import torch.nn as nn

class In(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = next(self.score_net.parameters()).device
        x = x.to(device)
        y = y.to(device)
        features = torch.cat([x, y], dim=1)
        return self.score_net(features)

class On(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = next(self.score_net.parameters()).device
        x = x.to(device)
        y = y.to(device)
        features = torch.cat([x, y], dim=1)
        return self.score_net(features)

class NextTo(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = next(self.score_net.parameters()).device
        x = x.to(device)
        y = y.to(device)
        features = torch.cat([x, y], dim=1)
        return self.score_net(features)

class OnTopOf(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = next(self.score_net.parameters()).device
        x = x.to(device)
        y = y.to(device)
        features = torch.cat([x, y], dim=1)
        return self.score_net(features)
        
class Near(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = next(self.score_net.parameters()).device
        x = x.to(device)
        y = y.to(device)
        features = torch.cat([x, y], dim=1)
        return self.score_net(features)

class Under(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = next(self.score_net.parameters()).device
        x = x.to(device)
        y = y.to(device)
        features = torch.cat([x, y], dim=1)
        return self.score_net(features)