import torch
import sys
import os
import random
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.utils import MNIST, train
from src.model import CNN
from src.optimizer import NoisySGD

import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Папка для сохранения моделей
CHECKPOINT_DIR = "data/checkpoints/exp2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Гиперпараметры
num_epochs = 10
learning_rate = 0.01
batch_size = 64
num_seeds = 5  # Количество различных запусков с разными сидом

# Загружаем данные
train_dataset, test_dataset, train_loader, test_loader = MNIST()

for seed in range(num_seeds):
    print(f"Running training with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Инициализация модели
    model = CNN().to(DEVICE)
    optimizer = NoisySGD(model.parameters(), lr=learning_rate, noise_std=0.01)
    criterion = nn.CrossEntropyLoss()

    # Обучение модели
    train_log, train_acc_log, val_log, val_acc_log = train(
        model, optimizer, train_loader, test_loader, criterion, num_epochs, DEVICE 
    )

    # Сохранение модели
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_seed_{seed}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model with seed {seed} saved at {checkpoint_path}")
