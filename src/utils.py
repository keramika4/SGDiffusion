import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

import torch.nn.functional as F

def MNIST():
    # Загрузка и подготовка данных
    transform = transforms.Compose([
        # transforms.Resize(32),  # Увеличиваем размер изображений до 32x32
        # transforms.Grayscale(3),  # Преобразуем в 3 канала
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))  # Нормализация для одного канала, дублируется для всех трех
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_dataset, test_dataset, train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

def train_epoch(model, optimizer, train_loader, criterion, device, prepare = None):
    model.train()
    for it, traindata in enumerate(train_loader):
        train_inputs, train_labels = traindata
        train_inputs = train_inputs.to(device) 
        train_labels = train_labels.to(device)
        train_labels = torch.squeeze(train_labels)

        model.zero_grad()        
        if prepare is not None:
            train_inputs = prepare(train_inputs, train_labels)

        output = model(train_inputs)
        loss = criterion(output, train_labels.long())
        loss.backward()
        optimizer.step()
def evaluate_loss_acc(loader, model, criterion, device):
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    for it, data in enumerate(loader):
        inputs, labels = data
        inputs = inputs.to(device) 
        labels = labels.to(device)
        labels = torch.squeeze(labels)

        output = model(inputs) # pay attention here!
        loss = criterion(output, labels.long())# + torch.norm(WW^T - I)
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct = pred == labels.byte()
        total_acc += torch.sum(correct).item() / len(correct)

    total = it + 1
    return total_loss / total, total_acc / total

def train(model, opt, train_loader, test_loader, criterion, n_epochs, \
          device, verbose=True, prepare = None):
    train_log, train_acc_log = [], []
    val_log, val_acc_log = [], []

    for epoch in range(n_epochs):
        train_epoch(model, opt, train_loader, criterion, device, prepare = prepare)
        train_loss, train_acc = evaluate_loss_acc(train_loader,
                                                  model, criterion,
                                                  device)
        val_loss, val_acc = evaluate_loss_acc(test_loader, model,
                                              criterion, device)

        train_log.append(train_loss)
        train_acc_log.append(train_acc)

        val_log.append(val_loss)
        val_acc_log.append(val_acc)

        if verbose:
             print (('Epoch [%d/%d], Loss (train/test): %.4f/%.4f,'+\
               ' Acc (train/test): %.4f/%.4f' )
                   %(epoch+1, n_epochs, \
                     train_loss, val_loss, train_acc, val_acc))

    return train_log, train_acc_log, val_log, val_acc_log

