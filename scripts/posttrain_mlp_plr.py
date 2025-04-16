import torch
import os
import sys
import random
import numpy as np
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.utils import MNIST, train, LOGGER
from src.model import MLP_PLR

import torch.nn as nn

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# Пути
CHECKPOINT_DIR = "data/checkpoints/exp_mlp_plr"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model.pth")
LOG_PATH = os.path.join(CHECKPOINT_DIR, "MLP_PLR_finetune_log")

# Гиперпараметры
num_epochs = 10
learning_rate = 0.001
batch_size = 64

# Датасеты
train_dataset, test_dataset, train_loader, test_loader = MNIST(batch_size=batch_size)

# Инициализация модели и загрузка весов
model = MLP_PLR(
    input_size=28*28,
    num_classes=10,
    hidden_dim=128,
    num_layers=2,
    embedding_type='periodic',
    d_embedding=128,
    n_frequencies=32,
    frequency_init_scale=1.0,
    activation='ReLU',
    dropout=0.0,
).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

# Замораживаем все параметры, кроме embedding.periodic.weight
for name, param in model.named_parameters():
    param.requires_grad = ('embedding.periodic.weight' in name)

# Оптимизатор только для разблокированных параметров
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Логгер
LOGGER.reset()
LOGGER.set_tick_step(1)

# Дообучение
for epoch in range(num_epochs):
    model.train()
    for x_batch, y_batch in tqdm(train_loader):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        LOGGER.update("loss", loss.item())

        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'embedding.periodic.weight' in name:
                    LOGGER.update(name, param.detach().cpu().numpy().copy())

    print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {loss.item():.4f}")

# Сохранение логов
LOGGER.save(LOG_PATH)
print(f"Finetune logs saved at {LOG_PATH}")
