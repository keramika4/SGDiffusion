import torch
import os
import sys
import random
import numpy as np
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.utils import MNIST, train, total_variation_loss_model, LOGGER
from src.model import MLP_PLR

import torch.nn as nn

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

# Папка для сохранения чекпоинтов и логов
CHECKPOINT_DIR = "data/checkpoints/exp_mlp_plr"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Гиперпараметры
num_epochs = 100
learning_rate = 0.01
batch_size = 64

alphas = [1e-4]  # Можешь добавить больше значений
print(f"Alphas: {alphas}")

train_dataset, test_dataset, train_loader, test_loader = MNIST(batch_size=batch_size)

for alpha in alphas:
    seed = 100
    print(f"\nRunning training with seed {seed} and alpha {alpha}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Инициализация модели
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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Регуляризатор
    regularizer = lambda model: alpha * total_variation_loss_model(model)

    LOGGER.reset()
    LOGGER.set_tick_step(1)

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in tqdm(train_loader):
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            if regularizer is not None:
                loss += regularizer(model)
            loss.backward()
            optimizer.step()

            LOGGER.update("loss", loss.item())

            # Логгирование весов PeriodicEmbeddings после каждого шага
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'frequencies' in name:
                        LOGGER.update("frequencies", param.detach().cpu().numpy().copy())

    # Сохраняем веса модели
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_alpha_{alpha:.0e}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model with alpha={alpha:.0e} saved at {checkpoint_path}")

    # Сохраняем логи
    log_path = os.path.join(CHECKPOINT_DIR, f"MLP_PLR_alpha_{alpha:.0e}_log")
    LOGGER.save(log_path)
    print(f"Logs saved at {log_path}")
