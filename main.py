# main_train.py
import os
from multiprocessing import freeze_support

if __name__ == "__main__":
    freeze_support()  # для Windows, безопасно и на macOS

    # main_train.py
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils.config import load_config
    from utils.seed import set_seed
    from data.loader import build_dataloaders
    from src.models.model_loader import (
        load_mobilenet_v2,
        apply_full_freeze,
        apply_partial_unfreeze_mobilenetv2,
        apply_full_finetune,
        count_params_millions
    )
    from src.training.train import Trainer
    import os
    
    # --------------------------
    # Конфигурация
    # --------------------------
    config_path = "configs/freeze.yaml"  # можно заменить на любой из freeze/partial/full
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --------------------------
    # Датасеты и DataLoader
    # --------------------------
    train_loader, val_loader = build_dataloaders(cfg)
    
    # --------------------------
    # Модели
    # --------------------------
    num_classes = len(cfg["data"]["classes"])
    
    # создаём отдельные экземпляры модели для каждого режима
    models_dict = {
        "Freeze": apply_full_freeze(load_mobilenet_v2(num_classes=num_classes, pretrained=True)),
        "Partial": apply_partial_unfreeze_mobilenetv2(load_mobilenet_v2(num_classes=num_classes, pretrained=True), unfreeze_last_n=2),
        "Full": apply_full_finetune(load_mobilenet_v2(num_classes=num_classes, pretrained=True))
    }
    
    # вывод параметров
    for name, model in models_dict.items():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = count_params_millions(model)
        print(f"{name}: trainable params = {trainable_params} / total params = {total_params:.2f}M")
    
    # --------------------------
    # Тренировка
    # --------------------------
    histories = {}
    for name, model in models_dict.items():
        print(f"\n=== Training {name} model ===")
        trainer = Trainer(
            model=model,
            device=device,
            lr=cfg["training"]["lr"],
            epochs=cfg["training"]["epochs"],
            train_loader=train_loader,
            val_loader=val_loader
        )
        history = trainer.fit()
        histories[name] = history
    
    # --------------------------
    # Построение графиков
    # --------------------------
    os.makedirs("outputs/plots", exist_ok=True)
    
    # Loss
    plt.figure(figsize=(8,6))
    for name, history in histories.items():
        plt.plot(history["train_loss"], label=f"{name} train")
        plt.plot(history["val_loss"], label=f"{name} val")
    plt.title("Loss curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/plots/loss_curves.png")
    plt.close()
    
    # Accuracy
    plt.figure(figsize=(8,6))
    for name, history in histories.items():
        plt.plot(history["train_acc"], label=f"{name} train")
        plt.plot(history["val_acc"], label=f"{name} val")
    plt.title("Accuracy curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("outputs/plots/accuracy_curves.png")
    plt.close()
    
    print("Training done. Plots saved to outputs/plots/")
    
