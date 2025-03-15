import torch
import sys
import os
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import CIFAR
from src.model import CIFAR_CNN,CNN

import torch.nn as nn

DEVICE = 'mps'

CHECKPOINT_DIR = "data/checkpoints/exp6"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_epoch(model, optimizer, train_loader, criterion, device):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_loss_acc(data_loader, model, criterion, device):
    model.eval()
    total_loss, total_correct = 0.0, 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


num_epochs = 20
learning_rate = 0.001
batch_size = 64
num_seeds = 3  # Количество различных запусков с разными сидом
num_checkpoints = 5  # Количество чекпоинтов
checkpoint_intervals = num_epochs // num_checkpoints

logging_tick_step = 1

# Загружаем данные
train_dataset, test_dataset, train_loader, test_loader = CIFAR()

for seed in range(num_seeds):
    print(f"Running training with seed {seed}")
    
    # Инициализация модели с одними и теми же весами
    torch.manual_seed(228)
    np.random.seed(228)
    random.seed(228)
    model = CIFAR_CNN().to(DEVICE)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        train_epoch(model, optimizer, train_loader, criterion, DEVICE)
        
        # Оценка модели
        train_loss, train_acc = evaluate_loss_acc(train_loader, model, criterion, DEVICE)
        val_loss, val_acc = evaluate_loss_acc(test_loader, model, criterion, DEVICE)
        
        print(f"Epoch [{epoch}/{num_epochs}], Loss (train/test): {train_loss:.4f}/{val_loss:.4f}, Acc (train/test): {train_acc:.4f}/{val_acc:.4f}")
        
        # Сохранение чекпоинта
        if epoch % checkpoint_intervals == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_seed_{seed}_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")