import torch
import sys
import os
import random
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.utils import MNIST, train, total_variation_loss_model
from src.model import CNN, MLP
from src.optimizer import NoisySGD

import torch.nn as nn

DEVICE = 'mps'

# Папка для сохранения моделей
CHECKPOINT_DIR = "data/checkpoints/exp2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Гиперпараметры
num_epochs = 10
learning_rate = 0.01
batch_size = 64

alphas = np.arange(1e-4, 1e-3, 1e-4)
print(alphas)

train_dataset, test_dataset, train_loader, test_loader = MNIST()

for alpha in alphas:
    seed = 100
    print(f"Running training with seed {seed} and alpha {alpha}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Инициализация модели
    model = MLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()


    regularizer = lambda model: alpha * total_variation_loss_model(model)
    # regularizer = None

    # Обучение модели
    train_log, train_acc_log, val_log, val_acc_log = train(
        model, optimizer, train_loader, test_loader, criterion, num_epochs, DEVICE , regularizer=regularizer
    )

    # Сохранение модели
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_alpha_{alpha}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model with alpha {alpha} saved at {checkpoint_path}")
