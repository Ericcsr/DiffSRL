import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self):
        super(InfoNCELoss,self).__init__()

    def forward(self,latent,latent_pred,latent_next):
        batch_size = latent.shape[0]

        neg_dot_prod = torch.mm(latent_pred,latent.t())
        neg_dists = -((latent_pred**2).sum(1).unsqueeze(1)- 2*neg_dot_prod+ (latent**2).sum(1).unsqueeze(0))
        idxs = np.arange(batch_size)
        neg_dists[idxs,idxs] = float('-inf')

        pos_dot_prod = (latent_next*latent_pred).sum(dim=1)
        pos_dists = -((latent_next**2).sum(1)-2*pos_dot_prod+(latent_pred**2).sum(1))
        pos_dists = pos_dists.unsqueeze(1)

        dists = torch.cat((neg_dists,pos_dists), dim=1)
        dists = F.log_softmax(dists,dim=1)
        loss = -dists[:,-1].mean()
        return loss