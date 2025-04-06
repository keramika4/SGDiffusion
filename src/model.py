import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class CNN(nn.Module):
    def __init__(self, k=1):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.layer1(x)  #;print(x.shape)
        x = self.layer2(x) #;print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x) #;print(x.shape)
        x = self.relu(x) #;print(x.shape)
        x = self.fc1(x) #;print(x.shape)
        x = self.relu1(x) #;print(x.shape)
        x = self.fc2(x) #;print(x.shape)
        return x
    
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super(CIFAR_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn512 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn64(self.conv1(x))))
        x = self.pool(F.relu(self.bn128(self.conv2(x))))
        x = F.relu(self.bn256(self.conv3(x)))
        x = self.pool(F.relu(self.bn256(self.conv4(x))))
        x = F.relu(self.bn512(self.conv5(x)))
        x = self.pool(F.relu(self.bn512(self.conv6(x))))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_size = 28*28, num_classes = 10, hidden_dim = 32, num_layers = 2):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_dim), nn.ReLU()]
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        return self.network(x)


class PeriodicEmbeddings(nn.Module):
    def __init__(self, d_in, d_embedding, n_frequencies=16, frequency_init_scale=1.0):
        super().__init__()
        self.frequencies = nn.Parameter(
            frequency_init_scale * torch.randn(d_in, n_frequencies)
        )
        self.linear = nn.Linear(d_in * 2 * n_frequencies, d_embedding)

    def forward(self, x):
        x_proj = 2 * math.pi * x.unsqueeze(-1) * self.frequencies
        x_pe = torch.cat([x_proj.sin(), x_proj.cos()], dim=-1)
        return self.linear(x_pe.view(x.size(0), -1))


def make_mlp(in_dim, out_dim, hidden_dim, num_layers, activation='ReLU', dropout=0.0):
    act_layer = getattr(nn, activation)
    layers = [nn.Linear(in_dim, hidden_dim), act_layer(), nn.Dropout(dropout)]
    for _ in range(num_layers):
        layers += [nn.Linear(hidden_dim, hidden_dim), act_layer(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class MLP_PLR(nn.Module):
    def __init__(
        self,
        input_size=28 * 28,
        num_classes=10,
        hidden_dim=128,
        num_layers=2,
        embedding_type='periodic',
        d_embedding=128,
        n_frequencies=32,
        frequency_init_scale=1.0,
        activation='ReLU',
        dropout=0.0,
    ):
        super().__init__()

        if embedding_type == 'periodic':
            self.embedding = PeriodicEmbeddings(
                d_in=input_size,
                d_embedding=d_embedding,
                n_frequencies=n_frequencies,
                frequency_init_scale=frequency_init_scale,
            )
            embedding_out_dim = d_embedding
        elif embedding_type == 'none':
            self.embedding = nn.Identity()
            embedding_out_dim = input_size
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")

        self.network = make_mlp(
            in_dim=embedding_out_dim,
            out_dim=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        return self.network(x)



class CNNLayerNorm(nn.Module):
    def __init__(self, k=1):
        super(CNNLayerNorm, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.LayerNorm([6, 28, 28]),  # LayerNorm после свертки
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm([6, 14, 14])  # LayerNorm после MaxPool2d
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.LayerNorm([16, 10, 10]),  # LayerNorm после свертки
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LayerNorm([16, 5, 5])  # LayerNorm после MaxPool2d
        )
        self.fc = nn.Linear(400, 120)
        self.ln1 = nn.LayerNorm(120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.ln2 = nn.LayerNorm(84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)
        self.ln3 = nn.LayerNorm(10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.ln2(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.ln3(x)
        return x
