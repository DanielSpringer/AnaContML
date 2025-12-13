from torch.utils.data import Dataset
import torch
import numpy as np
import h5py
import glob
from copy import deepcopy
from scipy.special import eval_legendre
import random

class Dataset_graph_InvPro(Dataset):
    def __init__(self, config, dataset, **kwargs):

        data_in = dataset[:,0,:config["omega_steps"]]
        data_target = dataset[:,1,:config["omega_steps"]]

        self.data_target = torch.tensor(data_target, dtype=torch.torch.float32)
        self.data_in = torch.tensor(data_in, dtype=torch.float32)
        
        self.n_nodes = config["n_nodes"]
        n_freq = self.data_in.shape[1]
        beta = 30 ### Later this needs to be dynamics
        iv = np.linspace(0, (2*n_freq+1)*np.pi/beta, config["omega_steps"])

        self.vectors = torch.tensor(np.load(config["PATH_VEC"]), dtype=torch.torch.float32)
        self.n_vectors = self.vectors.shape[0]

        # Nodes ultimately determine the number of vectors used.
        if self.n_vectors > self.n_nodes:
            self.n_vectors = self.n_nodes 
            self.vectors = self.vectors[:self.n_nodes]

        edge_index = torch.zeros((2, self.n_nodes**2))
        k = 0
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                edge_index[0, k] = i
                edge_index[1, k] = j
                k += 1
        self.edge_index = edge_index.long()

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        node_features = torch.zeros((self.n_nodes, 2*self.vectors.shape[1])) 
        for w in range(self.n_nodes):
            node_features[w,:] = torch.cat([self.vectors[w], self.data_in[idx]])

        sample = {}
        sample["node_feature"] = node_features
        sample["edge_index"] = self.edge_index
        sample["target"] = self.data_target[idx]
        sample["vectors"] = self.vectors
        return sample
