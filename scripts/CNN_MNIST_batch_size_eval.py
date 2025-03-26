

import torch
import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import CNN
from src.utils import MNIST

DEVICE = 'mps'  # Для Mac с чипом M1/M2
CHECKPOINT_DIR = "data/checkpoints/exp11"
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

# Параметры
# batch_sizes = [4, 8, 32, 64, 128, 256, 512, 1028] # на 128 уже не учится
# batch_sizes = [128, 256, 512, 768, 1024]
# batch_sizes = list(map( lambda x : int(x) ,np.arange(1,5)*1024))
# batch_sizes = list(map( lambda x : int(x) ,np.arange(10,50,10)*1024))
batch_sizes = list(map( lambda x : int(x) ,np.arange(50,100,5)*1024))
num_epochs = 10
learning_rate = 0.001

losses = {batch_size: [] for batch_size in batch_sizes}

for batch_size in batch_sizes:
        print(f"Running training with batch size {batch_size}")

        # Инициализация модели с одними и теми же весами
        torch.manual_seed(228)
        np.random.seed(228)
        random.seed(228)
        model = CNN().to(DEVICE)

        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()

        # Загружаем данные с текущим батчсайзом
        train_dataset, test_dataset, train_loader, test_loader = MNIST(batch_size=batch_size)

        for epoch in range(1, num_epochs + 1):
            train_epoch(model, optimizer, train_loader, criterion, DEVICE)
            
            # Оценка модели
            train_loss, train_acc = evaluate_loss_acc(train_loader, model, criterion, DEVICE)
            val_loss, val_acc = evaluate_loss_acc(test_loader, model, criterion, DEVICE)
            
            print(f"Epoch [{epoch}/{num_epochs}], Loss (train/test): {train_loss:.4f}/{val_loss:.4f}, Acc (train/test): {train_acc:.4f}/{val_acc:.4f}")
        
            # Сохраняем потери
            losses[batch_size].append(val_loss)

        # Сохранение чекпоинта
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_batch_{batch_size}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

# Строим график лосса для разных батчсайзов
plt.figure(figsize=(10, 6))
for batch_size, loss_values in losses.items():
    plt.plot(losses[batch_size], label=f'Batch size {batch_size}')

plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss for Different Batch Sizes')
plt.legend()
plt.grid(True)
plt.savefig('batch_size_loss.png')
plt.show()