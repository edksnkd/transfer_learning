import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


class Trainer:
    def __init__(self, model, device, lr, epochs, train_loader, val_loader=None):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader

        # оптимизатор получает ТОЛЬКО параметры с requires_grad=True
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )
        self.criterion = nn.CrossEntropyLoss()

    

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()                 
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def validate_epoch(self):
        if self.val_loader is None:
            return None, None

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        return total_loss / total, correct / total

    def fit(self):
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        early_stopping = EarlyStopping(patience=3)


        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            early_stopping.step(val_loss)

        
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)


            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if early_stopping.stop:
                print("Early stopping triggered")
                break

        return history
    
