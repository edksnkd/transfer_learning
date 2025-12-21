# Transfer Learning с MobileNetV2 на датасете CIFAR

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

В репозитории реализован пример **transfer learning** с использованием архитектуры **MobileNetV2** для задачи классификации изображений на подмножестве классов датасета **CIFAR-10**.
Поддерживаются режимы **Freeze**, **Partial Fine-tuning** и **Full Fine-tuning**.

---

## Структура проекта

```
transfer_curs/
│
├── configs/          # YAML-конфигурации режимов обучения
├── data/             # CIFAR-10 и загрузчики данных
├── src/
│   ├── models/       # Загрузка моделей и логика заморозки слоёв
│   └── training/     # Тренировочный цикл и класс Trainer
├── utils/            # Утилиты (config, seed и др.)
├── outputs/          # Графики и чекпойнты
├── test.py           # Проверка модели и числа обучаемых параметров
└── main_train.py     # Полный запуск обучения и визуализация
```

---

## Установка

```bash
git clone https://github.com/<your-username>/transfer_curs.git
cd transfer_curs
conda create -n finetune python=3.11
conda activate finetune
pip install torch torchvision matplotlib seaborn pyyaml
```

---

## Конфигурация

Обучение управляется YAML-файлами в директории `configs/`.

| Конфигурация   | Режим             | Описание                          |
| -------------- | ----------------- | --------------------------------- |
| `freeze.yaml`  | Freeze            | Обучается только классификатор    |
| `partial.yaml` | Partial Fine-tune | Размораживаются последние N слоёв |
| `full.yaml`    | Full Fine-tune    | Обучается вся модель              |

Пример `freeze.yaml`:

```yaml
inherits: "base.yaml"
training:
  mode: "freeze"
  lr: 1e-3
```

---

## Использование

### Проверка модели

```bash
python test.py --config configs/freeze.yaml
```

Скрипт:

* выводит общее число параметров;
* показывает количество обучаемых параметров для каждого режима;
* проверяет корректность прямого прохода (forward pass).

---

### Обучение модели

```bash
python main_train.py
```

В процессе:

* последовательно обучаются режимы **Freeze**, **Partial**, **Full**;
* сохраняются графики функции потерь и точности в `outputs/plots/`.

---

## Результаты

Пример результатов после 10 эпох обучения на 3 классах CIFAR-10:

| Режим   | Точность на валидации |
| ------- | --------------------- |
| Freeze  | ~69%                  |
| Partial | ~73%                  |
| Full    | ~87%                  |

Сохраняемые графики:

* `outputs/plots/loss_curves.png` — loss на train и val
* `outputs/plots/accuracy_curves.png` — accuracy на train и val

---

Таблица сравнений:
| Mode    | Best Val Accuracy | Train Accuracy | Overfitting | Time (sec) | Avg Time / Epoch (sec) | Epochs |
|---------|-------------------|----------------|-------------|------------|------------------------|--------|
| Freeze  | 0.937             | 0.930          | -0.007      | 684.4      | 68.4                   | 10     |
| Partial | 0.940             | 0.966          | 0.026       | 498.8      | 71.3                   | 7      |
| Full    | 0.943             | 0.948          | 0.005       | —          | —                      | 7      |


## Датасет

Используется датасет **CIFAR-10**.

Выбор классов осуществляется в конфигурационном файле:

```yaml
data:
  type: "cifar10"
  classes: [0, 1, 2]  # Подмножество классов
  batch_size: 32
  image_size: 32
```

Важно:
Количество выходных классов модели должно соответствовать длине списка `classes`.

---

## Лицензия

MIT License

---
