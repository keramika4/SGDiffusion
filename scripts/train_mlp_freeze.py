import torch
import os
import sys
import random
import numpy as np
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.utils import MNIST, LOGGER
from src.model import MLP_PLR
import torch.nn as nn

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

CHECKPOINT_DIR = "data/checkpoints/finetune_embedding_and_linear_frozen"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

num_epochs = 10
learning_rate = 0.001
batch_size = 64

train_dataset, test_dataset, train_loader, test_loader = MNIST(batch_size=batch_size)

seed = 100
print(f"\nRunning training with seed {seed} ")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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

# Заморозка указанных параметров
freeze_names = [
    "embedding.periodic.weight",
    "embedding.linear.weight",
    "embedding.linear.bias"
]

for name, param in model.named_parameters():
    param.requires_grad = all(f not in name for f in freeze_names)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

LOGGER.reset()
LOGGER.set_tick_step(1)

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
                if param.requires_grad:
                    LOGGER.update(name, param.detach().cpu().numpy().copy())

    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model.pth")
torch.save(model.state_dict(), checkpoint_path)
print(f"Model saved at {checkpoint_path}")

log_path = os.path.join(CHECKPOINT_DIR, f"log")
LOGGER.save(log_path)
print(f"Logs saved at {log_path}")
