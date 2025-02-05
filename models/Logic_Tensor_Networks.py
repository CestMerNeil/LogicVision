import torch
import torch.nn as nn
import ltn
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)
ltn_config = config["Logic_Tensor_Networks"]

class Logic_Tensor_Networks:
    def __init__(self, detector_output: dict, label_mapping: dict):
        """
        :param detector_output: 检测器输出字典 {
            'boxes': Tensor[batch, max_objs, 4],
            'classes': Tensor[batch, max_objs],
            'masks': Tensor[batch, max_objs, H, W],
            'scores': Tensor[batch, max_objs],
            ...其他字段
        }
        :param label_mapping: 标签字典 {0: 'background', 1: 'cup', ...}
        """
        # 初始化元数据
        self.label_mapping = label_mapping
        self.reverse_label = {v:k for k,v in label_mapping.items()}
        self.feat_dim = self._calculate_feature_dim(label_mapping)

        self.mlp_hidden_dims = ltn_config["mlp_hidden_dims"]
        self.mlp_dropout = ltn_config["mlp_dropout"]
        self.cnn_channels = ltn_config["cnn_channels"]
        self.cnn_unflatten = ltn_config["cnn_unflatten"]
        self.conv_kernel_size = ltn_config["conv_kernel_size"]
        self.pool_size = ltn_config["pool_size"]
        
        # 第一阶段：对象特征嵌入
        self.objects = self._process_detector_output(detector_output)
        
        # 第二阶段：构建关系网络
        self.predicates = nn.ModuleDict({
            'Near': self._build_mlp(
                input_dim=ltn_config["mlp_input_dim"],
                hidden_dims=self.mlp_hidden_dims,
                dropout=self.mlp_dropout
            ),
            #'TopOf': self._build_cnn(
            #    input_dim=ltn_config["cnn_input_dim"],
            #    channels=self.cnn_channels
            #)
        })
        
        # 存储原始检测结果
        self.raw_data = detector_output

    def _process_detector_output(self, data: dict) -> ltn.Constant:
        # 检查输入数据有效性
        required_keys = ['boxes', 'classes', 'masks', 'scores']
        if not all(k in data for k in required_keys):
            return ltn.Constant(torch.zeros(0, self.feat_dim))
        
        batch_size, max_objs = data['boxes'].shape[:2]
        obj_features = []
        
        for b in range(batch_size):
            batch_features = []
            for o in range(max_objs):
                # 跳过无效对象（class=0）
                if data['classes'][b,o].item() == 0:
                    continue
                    
                # 提取特征并拼接
                features = [
                    data['boxes'][b,o],
                    data['scores'][b,o].view(1),
                    self._encode_class(data['classes'][b,o].item()),
                    self._encode_mask(data['masks'][b,o])
                ]
                batch_features.append(torch.cat(features))
            
            if batch_features:
                obj_features.append(torch.stack(batch_features))
            else:
                obj_features.append(torch.zeros(0, self.feat_dim))
        
        # 处理全空批次的情况
        if not obj_features:
            return ltn.Constant(torch.zeros(0, self.feat_dim))
        
        return ltn.Constant(torch.stack(obj_features))
    
    def _calculate_feature_dim(self, label_mapping: dict) -> int:
        """动态计算特征总维度"""
        test_class = 1  # 非零的有效类别
        test_mask = torch.rand(64, 64)  # 假设掩码尺寸
        
        # 计算各部分维度
        box_dim = 4
        score_dim = 1
        class_dim = len(self._encode_class(test_class))
        mask_dim = len(self._encode_mask(test_mask))
        
        return box_dim + score_dim + class_dim + mask_dim

    def _encode_class(self, class_id: int) -> torch.Tensor:
        """修正后的类别编码"""
        return torch.nn.functional.one_hot(
            torch.tensor(class_id).long(),  # 确保为整数类型
            num_classes=len(self.label_mapping)
        ).float()

    def _encode_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """简化掩码编码"""
        return torch.cat([
            mask.flatten().float().mean().view(1),  # 平均像素值
            mask.flatten().std().view(1)            # 像素标准差
        ])

    def query(self, relation: str, subject: str, obj: str, threshold=None) -> list:

        if threshold is None:
            threshold = ltn_config["default_threshold"]

        subj_id = self.reverse_label[subject]
        obj_id = self.reverse_label[obj]
        
        results = []
        for b in range(self.raw_data['classes'].shape[0]):
            valid_mask = self.raw_data['classes'][b] != 0
            obj_indices = torch.where(valid_mask)[0].tolist()
            
            candidates = [
                (i,j) for i in obj_indices 
                for j in obj_indices 
                if i != j and
                self.raw_data['classes'][b,i] == subj_id and
                self.raw_data['classes'][b,j] == obj_id
            ]
            
            if candidates:
                pair_features = torch.stack([
                    torch.cat([self.objects.value[b,i], self.objects.value[b,j]]) 
                    for i,j in candidates
                ])
                
                # 关键修改：确保输出维度正确
                scores = self.predicates[relation](pair_features).view(-1)
                
                valid_pairs = [
                    (candidates[i], score.item()) 
                    for i, score in enumerate(scores) 
                    if score > threshold
                ]
            else:
                valid_pairs = []
            
            results.append({
                "batch": b,
                "pairs": valid_pairs,
                "metadata": {
                    "image_size": self.raw_data['image_size'][b].tolist(),
                    "num_objects": valid_mask.sum().item()
                }
            })
        
        return results

    # 神经网络构建工具方法
    @staticmethod
    def _build_mlp(input_dim, hidden_dims, dropout):
        layers = []
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(input_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            input_dim = h_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    @staticmethod
    def _build_cnn(self, input_dim, channels):
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Unflatten(1, self.cnn_unflatten),
            nn.Conv2d(1, channels[0], self.conv_kernel_size, padding=1),
            nn.MaxPool2d(self.pool_size),
            nn.Conv2d(channels[0], channels[1], self.conv_kernel_size),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[1], 1),
            nn.Sigmoid()
        )

if __name__ == "__main__":
    # 示例用法
    # 假设从检测器获取的输出（模拟YOLO输出）
    detector_data = {
        'boxes': torch.rand(2, 10, 4),          # 2 batches, 10 objects
        'classes': torch.tensor([
            [1,2,3,0,0,0,0,0,0,0], 
            [2,2,1,4,0,0,0,0,0,0]
        ]),      # 类别ID
        'masks': torch.rand(2, 10, 640, 640),   # 掩码
        'scores': torch.rand(2, 10),            # 置信度
        'image_size': torch.tensor([[640,640], [640,640]])
    }
    
    labels = {
        0: 'background',
        1: 'cup',
        2: 'bottle',
        3: 'table',
        4: 'chair'
    }

    # 初始化LTN
    ltn_engine = Logic_Tensor_Networks(detector_data, labels)
    
    # 执行查询
    print("Query results:", ltn_engine.query(
        relation="Near",
        subject="cup",
        obj="table",
        threshold=0.6
    ))