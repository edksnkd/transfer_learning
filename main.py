import argparse
import torch
from utils.seed import set_seed
from utils.config import load_config
from data.loader import build_dataloaders
from models.model_loader import (
    load_mobilenet_v2,
    apply_full_freeze,
    apply_partial_unfreeze_mobilenetv2,
    apply_full_finetune
)
from training.train import Trainer


def main():
    # ----------------
    # CLI
    # ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # ----------------
    # CONFIG
    # ----------------
    cfg = load_config(args.config)

    set_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------
    # DATA
    # ----------------
    train_loader, val_loader = build_dataloaders(cfg)

    num_classes = len(cfg["data"]["classes"])

    # ----------------
    # MODEL
    # ----------------
    model = load_mobilenet_v2(num_classes=num_classes)

    mode = cfg["training"]["mode"]
    if mode == "frozen":
        model = apply_full_freeze(model)
    elif mode == "partial":
        unfreeze_n = cfg["training"].get("unfreeze_last_n", 2)
        model = apply_partial_unfreeze_mobilenetv2(model, unfreeze_last_n=unfreeze_n)
    elif mode == "full":
        model = apply_full_finetune(model)
    else:
        raise ValueError(f"Unknown mode {mode}")

    # ----------------
    # TRAINER
    # ----------------
    trainer = Trainer(
        model=model,
        device=device,
        lr=cfg["training"]["lr"],
        epochs=cfg["training"]["epochs"],
        train_loader=train_loader,
        val_loader=val_loader
    )

    trainer.fit()


if __name__ == "__main__":
    main()
