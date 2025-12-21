
import os
import time
from multiprocessing import freeze_support

import torch
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.seed import set_seed
from data.loader import build_dataloaders
from src.models.model_loader import (
    load_mobilenet_v2,
    apply_full_freeze,
    apply_partial_unfreeze_mobilenetv2,
    apply_full_finetune,
    count_params_millions
)
from src.training.train import Trainer


if __name__ == "__main__":
    freeze_support() 

    # ==========================
    # CONFIG
    # ==========================
    config_path = "configs/partial.yaml"  # freeze / partial / full
    cfg = load_config(config_path)

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=== START TRAINING PIPELINE ===")
    print(f"Config file: {config_path}")
    print(f"Device: {device}")

    # ==========================
    # DATA
    # ==========================
    print("\n=== LOADING DATA ===")
    train_loader, val_loader = build_dataloaders(cfg)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    x, y = next(iter(train_loader))
    print(f"Sample batch shape: {x.shape}, labels shape: {y.shape}")

    # ==========================
    # MODELS
    # ==========================
    print("\n=== BUILDING MODELS ===")
    num_classes = len(cfg["data"]["classes"])

    models_dict = {
        "Freeze": apply_full_freeze(
            load_mobilenet_v2(num_classes=num_classes, pretrained=True)
        ),
        "Partial": apply_partial_unfreeze_mobilenetv2(
            load_mobilenet_v2(num_classes=num_classes, pretrained=True),
            unfreeze_last_n=cfg["training"].get("unfreeze_layers", 2)
        ),
        "Full": apply_full_finetune(
            load_mobilenet_v2(num_classes=num_classes, pretrained=True)
        )
    }

    for name, model in models_dict.items():
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = count_params_millions(model)
        print(f"{name}: trainable = {trainable_params}, total = {total_params:.2f}M")

    print("=== MODELS READY ===")

    # ==========================
    # TRAINING
    # ==========================
    histories = {}

    for name, model in models_dict.items():
        print(f"\n=== TRAINING {name.upper()} MODEL ===")
        print(f"LR: {cfg['training']['lr']}")
        print(f"Epochs: {cfg['training']['epochs']}")

        trainer = Trainer(
            model=model,
            device=device,
            lr=cfg["training"]["lr"],
            epochs=cfg["training"]["epochs"],
            train_loader=train_loader,
            val_loader=val_loader
        )

        start_time = time.time()
        history = trainer.fit()
        total_time = time.time() - start_time

        print(f"=== FINISHED {name.upper()} ===")
        print(f"Total time: {total_time:.2f} sec")
        print(f"Avg time / epoch: {total_time / len(history['train_loss']):.2f} sec")
        print(
            f"Best Val Acc: {max(history['val_acc']):.4f} | "
            f"Final Train Acc: {history['train_acc'][-1]:.4f}"
        )

        histories[name] = {
            **history,
            "time": total_time,
            "epochs": len(history["train_loss"])
        }

    # ==========================
    # PLOTS
    # ==========================
    print("\n=== PLOTTING RESULTS ===")
    os.makedirs("outputs/plots", exist_ok=True)

    # Loss curves
    plt.figure(figsize=(8, 6))
    for name, history in histories.items():
        plt.plot(history["train_loss"], label=f"{name} train")
        plt.plot(history["val_loss"], label=f"{name} val")
    plt.title("Loss curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/loss_curves.png")
    plt.close()

    # Accuracy curves
    plt.figure(figsize=(8, 6))
    for name, history in histories.items():
        plt.plot(history["train_acc"], label=f"{name} train")
        plt.plot(history["val_acc"], label=f"{name} val")
    plt.title("Accuracy curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/plots/accuracy_curves.png")
    plt.close()

    print("Training finished.")
    print("Plots saved to outputs/plots/")
