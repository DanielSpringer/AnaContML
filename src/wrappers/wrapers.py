import sys
sys.path.append('/gpfs/data/fs71925/dspringer1/Projects/AnaContML/src')
import torch 
from torch import nn
import json
import pytorch_lightning as L
from models import models
import gc
import numpy as np
import functools
from collections import OrderedDict
from torch.func import functional_call, vmap, jacrev

class model_wraper_gnn(L.LightningModule):

    def __init__(self, config):
        super().__init__()
        from torch_geometric.nn import MessagePassing, global_mean_pool
        module = __import__("models.models", fromlist=['object'])
        self.model = getattr(module, config["MODEL_NAME"])(config)
        self.criterion_mse = nn.MSELoss()
        self.config = config
        self.val_pred = []
        self.val_loss = []

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch).float()
        target = batch["target"][:].float()
        loss = self.criterion_mse(pred / abs(torch.sum(target, axis=1))[:, None], target / abs(torch.sum(target, axis=1))[:, None])
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch).float()
        target = batch["target"][:].float()
        loss = self.criterion_mse(pred / abs(torch.sum(target, axis=1))[:, None], target / abs(torch.sum(target, axis=1))[:, None])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def custom_validation(self, batch):
        pred = self.forward(batch).float()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        return optimizer
    
    def load_model_state(self, PATH):
        checkpoint = torch.load(PATH, map_location='cuda:0')
        self.model.load_state_dict(checkpoint['state_dict'])

