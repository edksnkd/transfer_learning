import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.utils.data import random_split


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

    # img_size = data_cfg.get("img_size", 32)
    # batch_size = data_cfg.get("batch_size", 64)


    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    if data_cfg["type"] == "cifar10":
        train_base = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        test_base = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    elif data_cfg["type"] == "mnist":
        train_base = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
        test_base = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {data_cfg['type']}")

    # class selection
    classes = data_cfg["classes"]
    max_per_class = data_cfg.get("max_per_class", None)

    # train_ds = select_classes(train_base, classes, max_per_class)
    # test_ds  = select_classes(test_base, classes, max_per_class)

    # train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    # test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    full_train = select_classes(
    train_base,
    class_ids=data_cfg["classes"],
    max_per_class=data_cfg.get("max_per_class")
)

    val_ratio = data_cfg.get("val_ratio", 0.2)

    val_size = int(len(full_train) * val_ratio)
    train_size = len(full_train) - val_size

    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42))
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2)
    )

    return train_loader, val_loader


