import sys
sys.path.append('/gpfs/data/fs71925/dspringer1/Projects/AnaContML/')

import torch 
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool
import copy
import pytorch_lightning as pl

class Swish(nn.Module):
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class GNN_1_Layer(MessagePassing):
    def __init__(self, config):
        
        super(GNN_1_Layer, self).__init__(node_dim=-2, aggr='mean')
        
        message_in_dim = config["message_in_dim"]
        message_hidden_dim = config["message_hidden_dim"]
        message_out_dim = config["message_out_dim"]
        update_in_dim = config["update_in_dim"]
        update_hidden_dim = config["update_hidden_dim"]
        update_out_dim = config["update_out_dim"]
        n_nodes = config["n_nodes"]

        self.message_net = nn.Sequential(
            nn.Linear(message_in_dim, message_hidden_dim),
            Swish(),
            nn.BatchNorm1d(int(n_nodes*n_nodes)),
            nn.Linear(int(message_hidden_dim), int(message_hidden_dim)),
            Swish(),
            nn.BatchNorm1d(int(n_nodes*n_nodes)),
            nn.Linear(int(message_hidden_dim), int(message_hidden_dim)),
            Swish(),
            nn.BatchNorm1d(int(n_nodes*n_nodes)),
            nn.Linear(int(message_hidden_dim), message_out_dim),
            Swish()
        )
        self.update_net = nn.Sequential(
            nn.Linear(update_in_dim, int(update_hidden_dim)),
            Swish(),
            nn.Linear(int(update_hidden_dim), int(update_hidden_dim)),
            Swish(),
            nn.Linear(int(update_hidden_dim), int(update_hidden_dim)),
            Swish(),
            nn.Linear(int(update_hidden_dim), update_out_dim),
            Swish()
        )

    def forward(self, x, edge_index, v):
        """ Propagate messages along edges """
        propagate = self.propagate(edge_index, x=x, v=v)
        # print("PROPAGATE")
        # print("Node Features 2xiv (ImG | Ve):" , x.shape)
        # print("Vectors:" , v.shape)
        return propagate

    def message(self, x_i, x_j, x):
        """ Message creation over neighbours 
        x_i: node_i [available/incoming messages for each node i] (including own)
                DIM[x_i]: [batch, n_nodes**2, node_feature]

        x_j: node_j [messages sent by node j] (N1 identical lines if node1 is connect to N1 neighbours)
                DIM[x_j]: [batch, n_nodes**2, node_feature]

        features: Concatenation of x_i and x_j generates all possible combinations of messages 
                DIM[features]: [batch, n_nodes**2, 2*node_feature]
            First Layer:         [vector_i | ImG] | [vector_j | ImG]
            Follow up Layers:   [FeatureVector_i] | [FeatureVector_j]

        message_net(features): creates local messages for all the concatenated vectors (i.e. all combination)
        """
        features = torch.cat((x_i, x_j[:, :int(x_j.shape[1])]), dim=-1)
        message = self.message_net(features)
        # print(x_i.shape)
        # print(x_j.shape)
        # print(x_j[:, :int(x_j.shape[1])].shape)
        # print(x_i[0,:,0:4])
        # print(x_j[0,:,0:4])
        # print(features.shape)
        # print("MESSAGE")
        # print(message.shape)
        # print("  ---------------  ")
        return message

    def update(self, agg_message, x, v):
        """ Node update 
        v: Original vectors
            DIM[v]: [batch, n_nodes, omega_steps]

        agg_message: Node-wise output of messageNet
            DIM[agg_message]: [batch, n_nodes, message_out_dim]

        x: Node-wise features (node_features) before messageNet
            DIM[x]: [batch, n_nodes, node_feature]

        x += update_net(cat[v,x,message]): 
            > The concatenation of [message | node_feature | vector] is reminiscent (concat != summation) 
              of a ResNet with respect to message_net (x is the residual and v is the super residual that never changes)
            > update_net output is identical in dimension as x to be again added to the residual x (ResNet with respect to update_net)
        """
        x += self.update_net(torch.cat((v, x, agg_message), dim=-1))
        # print("v: ", v.shape)
        # print("x: ", x.shape)
        # print("Message: ", agg_message.shape)
        # print(v[0,:,0:4])
        # print(x[0,:,0:4])
        # print(agg_message[0,:,0:4])
        # print(torch.cat((v, x, agg_message), dim=-1).shape)
        # print("UPDATE")
        # print("UpdateNet Output: ", self.update_net(torch.cat((v, x, agg_message), dim=-1)).shape)
        # print("  ---------------  ")
        return x

class GNN_1_base(torch.nn.Module):
    def __init__(self, config):
        super(GNN_1_base, self).__init__()
        self.config = config

        self.out_dim = config["out_dim"]
        self.message_in_dim = config["message_in_dim"]                           # 2 Elements: neighbouring feature (v, G)
        self.message_hidden_dim = config["message_hidden_dim"]
        self.message_out_dim = config["message_out_dim"]

        if "update_in_dim" in config:
            self.update_in_dim = config["update_in_dim"]  
        else:
            self.update_in_dim = config["message_out_dim"] + int(config["message_in_dim"]) # 3 Elements: agg message, local v, local feature (v, G)
        
        self.update_hidden_dim = config["update_hidden_dim"]
        self.update_out_dim = config["update_out_dim"] # config["omega_steps"] + 1*config["omega_steps"]
        self.nr_coefficients = config["nr_coefficients"]
        self.hidden_layer = config["hidden_layer"]
        self.pre_pool_hidden_dim = config["pre_pool_hidden_dim"]
        self.pre_pool_out_dim = config["pre_pool_out_dim"]
        self.post_pool_hidden_dim = config["post_pool_hidden_dim"]
        self.post_pool_out_dim = config["nr_coefficients"]
        self.n_nodes = config["n_nodes"]

        # in_features have to be of the same size as out_features for the time being
        self.green_gnn = torch.nn.ModuleList(
            modules=[GNN_1_Layer(config) for _ in range(self.hidden_layer)]
        )

        self.head_pre_pool = nn.Sequential(
            nn.Linear(self.update_out_dim, int(self.pre_pool_hidden_dim)),
            Swish(),
            nn.Linear(int(self.pre_pool_hidden_dim), int(self.pre_pool_hidden_dim * 1)),
            Swish(),
            nn.Linear(int(self.pre_pool_hidden_dim * 1), self.pre_pool_hidden_dim),
            Swish(),
            nn.Linear(self.pre_pool_hidden_dim, self.pre_pool_out_dim))

        self.head_post_pool = nn.Sequential(
            nn.Linear(self.pre_pool_out_dim, self.post_pool_hidden_dim),
            Swish(),
            nn.Linear(self.post_pool_hidden_dim, 1))
            # nn.Linear(self.post_pool_hidden_dim, self.nr_coefficients))

    def forward(self, data): #, G):
        edge_index = data["edge_index"][0]
        x = data["node_feature"][:]
        x1 = data["vectors"][:]
        
        x2 = copy.deepcopy(x)
        
        for i in range(self.hidden_layer):
            x2 = self.green_gnn[i](x2, edge_index, v=x1)
        x2 = self.head_pre_pool(x2)
        # batch = torch.zeros(x2.size(1), dtype=torch.long, device=x2.device)
        # x2 = global_mean_pool(x2, batch)
        coefficients = self.head_post_pool(x2)

        x3 = torch.zeros((x1.shape[0],x1.shape[2]), device=x2.device, dtype=torch.float64)
        for b in range(0, coefficients.shape[0]):
            for n in range(0, coefficients.shape[1]):
                x3[b,:] += x1[b,n,:] * coefficients[b,n,0]
        # VALIDATION
        return x3, x1, coefficients
        # TRAINING
        # return x3

    
