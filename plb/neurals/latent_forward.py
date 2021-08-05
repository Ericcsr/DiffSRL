import torch
import torch.nn as nn

class ForwardModel(nn.Module):
    def __init__(self, latent_dim, action_dim=0):
        super().__init__()
        self.latent_dim = latent_dim
        hidden_size = 64
        self.model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim)
        )

    def forward(self, z, a):
        x = torch.cat((z, a), dim=-1)
        x = self.model(x)
        return x

class InverseModel(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.model = nn.Sequential(
            nn.Linear(2 * latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, z, z_next):
        x = torch.cat((z, z_next), dim=1)
        return self.model(x)