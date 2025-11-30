import torch
import torch.nn as nn
import torchvision.models as models


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def count_params_millions(model):
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def get_mobilenetv2_blocks(model):
    """
    Возвращает список bottleneck-блоков MobileNetV2.
    model.features — это nn.Sequential, где:
      - features[0] — первый Conv
      - features[1:] — инвертированные bottleneck-блоки
    """
    blocks = list(model.features.children())
    return blocks  # полный список; bottleneck'и начинаются с index=1


# =========================
# ОСНОВНАЯ ФУНКЦИЯ ЗАГРУЗКИ
# =========================

def load_mobilenet_v2(num_classes, width_mult=1.0, pretrained=True):
    """
    Загружает MobileNetV2, заменяет классификационную голову.
    Параметры:
        num_classes  – число классов
        width_mult   – уменьшение/увеличение размеров слоёв (0.5, 0.75, 1.0)
        pretrained   – загрузка предобученных весов
    """
    model = models.mobilenet_v2(pretrained=pretrained, width_mult=width_mult)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# =========================
# РЕЖИМ 1 — ПОЛНАЯ ЗАМОРОЗКА
# =========================

def apply_full_freeze(model):
    """
    Заморозка всех слоёв, кроме классификационной головы.
    """
    for p in model.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True

    return model


# =========================
# РЕЖИМ 2 — ЧАСТИЧНЫЙ FINE-TUNING
# =========================

def apply_partial_unfreeze_mobilenetv2(model, unfreeze_last_n=2):
    """
    Разморозка последних N bottleneck-блоков MobileNetV2 + головы.
    unfreeze_last_n — число последних блоков.
    """
    # сначала всё фризим
    for p in model.parameters():
        p.requires_grad = False

    blocks = get_mobilenetv2_blocks(model)
    total_blocks = len(blocks)

    # индексы размораживаемых блоков (с конца)
    start_idx = max(0, total_blocks - unfreeze_last_n)
    blocks_to_unfreeze = blocks[start_idx:]

    for block in blocks_to_unfreeze:
        for p in block.parameters():
            p.requires_grad = True

    # и классификационная голова
    for p in model.classifier.parameters():
        p.requires_grad = True

    return model


# =========================
# РЕЖИМ 3 — ПОЛНЫЙ FINE-TUNING
# =========================

def apply_full_finetune(model):
    """
    Полная разморозка всех слоёв.
    """
    for p in model.parameters():
        p.requires_grad = True

    return model
