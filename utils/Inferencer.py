# utils/Inferencer.py
import tomllib
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from models import LTN, OneFormer_Extractor


def inference():
    # 读取配置文件
    config_path = Path("config.toml")
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    infer_config = config["Inferencer"]
    image_path = infer_config["image_path"]
    weights_path = infer_config["weights_path"]

    # 由于 OneFormer 不支持 .to()，因此此处均在 CPU 上运行
    device = torch.device("cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # 初始化 OneFormer 提取器（不调用 .to()）
    feature_extractor = OneFormer_Extractor()
    with torch.no_grad():
        detector_output = feature_extractor.predict(img_tensor)
    # 定义类别映射（需要与训练时一致）
    labels = {0: 'background', 1: 'cup', 2: 'bottle', 3: 'table', 4: 'chair'}

    ltn_model = LTN(detector_output, labels)
    ltn_model.predicates.to(device)
    ltn_model.eval()

    # 加载训练好的权重
    state_dict = torch.load(weights_path, map_location=device)
    ltn_model.predicates.load_state_dict(state_dict)
    print("Loaded weights from", weights_path)

    # 查询 “cup Near table” 关系的候选对
    query_results = ltn_model.query("Near", "cup", "table")
    print("Inference query results:")
    for result in query_results:
        print(result)

if __name__ == "__main__":
    inference()