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
            nn.Linear(input_dim * 2, 256),   # 增大输入层到 512
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),                 # 加入 Dropout，防止过拟合

            nn.Linear(256, 128),             # 增加中间层宽度
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),

            nn.Linear(128, 64),              # 保持一层较小的维度
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(64, 1),
            nn.Sigmoid()                     # 用于概率输出
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
            nn.Linear(input_dim * 2, 512),   # 增大输入层到 512
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),                 # 加入 Dropout，防止过拟合

            nn.Linear(512, 256),             # 增加中间层宽度
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),

            nn.Linear(256, 128),             # 增加隐藏层
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(128, 64),              # 保持一层较小的维度
            nn.LeakyReLU(negative_slope=0.01),

            nn.Linear(64, 1),
            nn.Sigmoid()                     # 用于概率输出
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