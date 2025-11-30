import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def select_classes(dataset, class_ids, max_per_class=None):
    idxs = []
    for cls in class_ids:
        cls_idxs = [i for i, (_, y) in enumerate(dataset) if y == cls]
        if max_per_class is not None:
            cls_idxs = cls_idxs[:max_per_class]
        idxs.extend(cls_idxs)
    return Subset(dataset, idxs)

def build_dataloaders(cfg):
    data_cfg = cfg["data"]

    img_size = data_cfg.get("img_size", 32)
    batch_size = data_cfg.get("batch_size", 64)

    # transforms
    train_tfms = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_tfms = transforms.Compose([
        transforms.ToTensor(),
    ])

    if data_cfg["type"] == "cifar10":
        train_base = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tfms)
        test_base = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfms)
    elif data_cfg["type"] == "mnist":
        train_base = datasets.MNIST(root="./data", train=True, download=True, transform=train_tfms)
        test_base = datasets.MNIST(root="./data", train=False, download=True, transform=test_tfms)
    else:
        raise ValueError(f"Unknown dataset: {data_cfg['type']}")

    # class selection
    classes = data_cfg["classes"]
    max_per_class = data_cfg.get("max_per_class", None)

    train_ds = select_classes(train_base, classes, max_per_class)
    test_ds  = select_classes(test_base, classes, max_per_class)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
