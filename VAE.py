import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load MNIST CSV
# -----------------------------
df1 = pd.read_csv("mnist_train.csv")
df2 = pd.read_csv("mnist_test.csv")
df = pd.concat([df1, df2], axis=0, ignore_index=True)

# Normalize + convert to tensor
features = torch.tensor((df.iloc[:, 1:].values / 255.0).astype(np.float32))
labels = torch.tensor(df.iloc[:, 0].values, dtype=torch.long)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# -----------------------------
# Dataset
# -----------------------------
class DataPipeline(Dataset):
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]

ds = DataPipeline(features)
dl = DataLoader(ds, batch_size=100, shuffle=True)

# -----------------------------
# VAE Model
# -----------------------------
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Latent
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# VAE Loss
# -----------------------------
def loss_fn(x, x_hat, mean, logvar):
    recon = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon + kld

# -----------------------------
# Training Loop
# -----------------------------
history = []
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for batch in dl:
        batch = batch.to(device)
        optimizer.zero_grad()
        x_hat, mean, logvar = model(batch)
        loss = loss_fn(batch, x_hat, mean, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    history.append(total_loss) 
    print(f"Epoch {epoch+1}/{epochs}, Loss = {total_loss/len(ds)}")
