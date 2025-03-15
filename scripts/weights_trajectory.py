import torch
import sys
import os
import random
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.utils import MNIST, train, LOGGER
from src.model import CNN, CNNLayerNorm
from src.optimizer import NoisySGD

import torch.nn as nn

DEVICE = 'mps'

# Папка для сохранения моделей
CHECKPOINT_DIR = "data/checkpoints/exp3"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

num_epochs = 5
learning_rate = 0.01
batch_size = 64
num_seeds = 1  # Количество различных запусков с разными сидом

logging_tick_step = 1

# Загружаем данные
train_dataset, test_dataset, train_loader, test_loader = MNIST()

for seed in range(num_seeds):
    print(f"Running training with seed {seed}")
    
    # Инициализация модели c одними и теми же весами
    torch.manual_seed(228)
    np.random.seed(228)
    random.seed(228)
    model = CNN().to(DEVICE)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    optimizer = NoisySGD(model.parameters(), lr=learning_rate, noise_std=0.0)
    criterion = nn.CrossEntropyLoss()

    # Обучение модели
    train_log, train_acc_log, val_log, val_acc_log = train(
        model, optimizer, train_loader, test_loader, criterion, num_epochs, DEVICE, 
        verbose = True, 
        logging_tick_step=logging_tick_step ,
        logging = True
    )

    # Сохранение модели
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_seed_{seed}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model with seed {seed} saved at {checkpoint_path}")

    LOGGER.save(filename = f'{CHECKPOINT_DIR}/MNIST_CNN_{logging_tick_step}tick_steps_{num_epochs}epochs_({seed}_{num_seeds})seeds')