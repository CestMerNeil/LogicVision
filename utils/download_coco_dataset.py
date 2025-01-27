from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch

def custom_collate_fn(batch):
    """
    自定义 collate_fn，确保批次中的图像和掩码可以被正确堆叠。
    """
    images = []
    masks = []

    for img, mask in batch:
        images.append(img)
        masks.append(mask)

    # 将图像和掩码分别堆叠为张量
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)

    return images, masks

def main():
    # 下载并加载 Pascal VOC 数据集
    dataset = VOCSegmentation(
        root="/Users/neil/Code/LogicVision/data",
        year="2012",
        image_set="val",
        download=True
    )

if __name__ == "__main__":
    main()