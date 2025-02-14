import torch
import torch.nn as nn
import ltn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class BaseNet(nn.Module):
    def __init__(self, input_dim):
        super(BaseNet, self).__init__()
        self.input_dim = input_dim
        self.fc_in = nn.Linear(input_dim, 128)

        self.conv1 = GCNConv(128, 128)
        self.conv2 = GCNConv(128, 64)

        self.fc_out = nn.Linear(64 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, combined):
        subj = combined[:, :self.input_dim]
        obj = combined[:, self.input_dim:]

        outputs = []
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=subj.device)

        for i in range(combined.shape[0]):
            node_features = torch.stack([subj[i], obj[i]], dim=0)
            node_features = F.relu(self.fc_in(node_features))
            node_features = F.relu(self.conv1(node_features, edge_index))
            node_features = F.relu(self.conv2(node_features, edge_index))
            combined_nodes = torch.cat([node_features[0], node_features[1]], dim=-1)
            score = self.fc_out(combined_nodes)
            outputs.append(score)
        outputs = torch.stack(outputs, dim=0)
        return self.sigmoid(outputs)

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
        net = nn.Sequential(
            # 第一层卷积：输入 (batch, 1, 2, input_dim)
            # 使用 padding 保持宽度尽可能不变
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(2, 3), padding=(0, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),

            # 第二层卷积：kernel_size=(1,2)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),

            # 第三层卷积：kernel_size=(1,2)
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),

            # 新增第四层卷积
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),

            # 新增第五层卷积
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.3),

            # 使用自适应池化保证输出固定为 (1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # 最终展平成 (batch, 64)
            
            nn.Linear(64, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        super().__init__(model=net)
        self.net = net
        self.input_dim = input_dim
    
    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1)
        cnn_input = combined.view(-1, 1, 2, self.input_dim)
        return self.net(cnn_input)

class Under(ltn.Predicate):
    def __init__(self, input_dim):
        net = BaseNet(input_dim)
        super().__init__(model=net)
        self.net = net

    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        combined = torch.cat([subj_tensor, obj_tensor], dim=-1)
        return self.net(combined)

