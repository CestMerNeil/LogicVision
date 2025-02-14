import torch
import torch.nn as nn
import ltn

class BaseNet(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layer=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, subj, obj):
        batch_size = subj.size(0)
        subj_emb = self.proj(subj).unsqueeze(1)
        obj_emb = self.proj(obj).unsqueeze(1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, subj_emb, obj_emb], dim=1)
        x = x + self.pos_embedding
        x = x.transpose(0, 1)
        encoded = self.transformer_encoder(x)
        cls_output = encoded[0]
        out = self.fc(cls_output)
        return out

class LTNBaseNet(ltn.Predicate):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layer=2, dropout=0.1):
        net = BaseNet(input_dim, d_model, nhead, num_layer, dropout)
        super().__init__(model=net)
        self.net = net

    def forward(self, subj: ltn.Variable, obj: ltn.Variable) -> torch.Tensor:
        device = next(self.net.parameters()).device
        subj_tensor = subj.value.to(device)
        obj_tensor = obj.value.to(device)
        return self.net(subj_tensor, obj_tensor)

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
        
class Near(LTNBaseNet):
    def __init__(self, input_dim):
        super().__init__(input_dim)

class Under(LTNBaseNet):
    def __init__(self, input_dim):
        super().__init__(input_dim)

