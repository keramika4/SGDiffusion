import torch
import sys
import os
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import MNIST, train, LOGGER
from src.model import CNN
from src.optimizer import NoisySGD

import torch.nn as nn

DEVICE = 'mps'

# Папка для сохранения моделей
CHECKPOINT_DIR = "data/checkpoints/exp5"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

num_epochs = 1

learning_rate = 0.1

batch_sizes = [1 , 4, 8, 16, 32, 64]
# batch_sizes = [64] 

logging_tick_step = 1


for batch_size in batch_sizes:
    print(f"Running training with batch size {batch_size}")
    train_dataset, test_dataset, train_loader, test_loader = MNIST(batch_size=batch_size, sample_size=100*batch_size)
    
    torch.manual_seed(228)
    np.random.seed(228)
    random.seed(228)

    model = CNN().to(DEVICE)
    optimizer = NoisySGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_log, train_acc_log, val_log, val_acc_log = train(
        model, optimizer, train_loader, test_loader, criterion, num_epochs, DEVICE, 
        verbose = True, 
        logging_tick_step=logging_tick_step ,
        gradient_logging=True
    )

    LOGGER.save(filename = f'{CHECKPOINT_DIR}/MNIST_CNN_{logging_tick_step}tick_steps_{num_epochs}epochs_{batch_size}batch_size')