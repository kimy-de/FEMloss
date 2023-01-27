import torch
import torch.nn as nn
import numpy as np

class FEMloss(nn.Module):
    def __init__(self, mmat, device):
        super(FEMloss, self).__init__()

        """
        mmat: mass matrix (N,N) - scipy sparse matrix
        device: 'cpu', 'gpu', etc.
        M: mass matrix (N,N) - torch sparse matrix
        """

        values = torch.FloatTensor(mmat.data).to(device)
        inds = torch.LongTensor(np.vstack((mmat.tocoo().row, mmat.tocoo().col))).to(device)
        self.M = torch.sparse.FloatTensor(inds, values, torch.Size(mmat.shape)).to(device)

    def forward(self, z):  

        """
        z: a bunch of vectors (B,N) 
        B: batch size
        Mz: Mz
        err: M-norm metric defined as sqrt((z.T)Mz)
        
        return: averaged M-norm error
        """
        Mz = torch.mm(self.M,z.T)
        err = torch.sqrt(torch.sum(z.T*Mz, dim=0))
        return torch.mean(err)