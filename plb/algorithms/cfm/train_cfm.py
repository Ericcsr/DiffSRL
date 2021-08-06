import numpy as np
import torch
from ...neurals.autoencoder import PCNEncoder, PCNDecoder
from ...neurals.latent_forward import ForwardModel
from ...neurals.pcdataloader import ChopSticksDataset
from .cpc_loss import InfoNCELoss
import argparse
from torch.utils.data import DataLoader

device = torch.device('cuda:0')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int,default=32)
args = parser.parse_args()

dataset = ChopSticksDataset()
dataloader = DataLoader(dataset,batch_size = args.batch_size)

n_particles = dataset.n_particles
n_actions = dataset.n_actions
latent_dim = 1024

encoder = PCNEncoder(
    state_dim=3,
    latent_dim=latent_dim)

forward_model = ForwardModel(
    latent_dim=latent_dim,
    action_dim = n_actions)

loss_fn = InfoNCELoss()

optimizer = torch.optim.Adam([encoder.parameters(),forward_model.parameters()],lr=0.0001)

def train(encoder,forward_model,optimizer,dataloader,loss_fn):
    total_loss = 0
    batch_cnt = 0
    for state, target, action in dataloader:
        optimizer.zero_grad()
        latent = encoder(state)
        latent_pred = forward_model(latent, action)
        latent_next = encoder(target)
        loss = loss_fn(latent, latent_pred, latent_next)
        total_loss += float(loss)
        loss.backward()
        optimizer.step()
    return total_loss/batch_cnt


for iter in range(args.num_iters):
    loss = train(encoder,forward_model,optimizer,dataloader, loss_fn)
    print("Iteration:",iter,"Loss:",loss)

torch.save(encoder.state_dict(),'pretrain_model/weight_cfm.pth')
