import torch
import torch.nn as nn
import ltn

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
    
    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        # 提取输入张量
        subj_tensor = subj.value  # shape: [15, 1, 5]
        obj_tensor = obj.value    # shape: [1, 1, 5]

        # 显式扩展 obj_tensor 以匹配 subj_tensor 的形状
        obj_tensor = obj_tensor.expand_as(subj_tensor)  # shape: [15, 1, 5]

        # 合并输入（在最后一个维度拼接）
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1)  # shape: [15, 1, 10]
        combined = combined.view(-1, 10)  # 展平为 [15, 10]

        # 输入模型并返回结果
        return self.net(combined)  # shape: [15, 1]

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
        
class Near(ltn.Predicate):
    def __init__(self, input_dim):
        # Define the neural network model for the Near predicate.
        net = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
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