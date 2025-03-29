from dgl.nn import GraphConv, HeteroGraphConv, SAGEConv

import torch
import torch.nn as nn
import torch.nn.functional as F

# 构建模型

class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, n_classes, n_layer):
        super().__init__()
        self.n_layer = n_layer
        self.gcnlayer = torch.nn.ModuleList()
        self.batchnorms = torch.nn.ModuleList()

        # Input Layer
        self.gcnlayer.append(SAGEConv(in_feats, hid_feats, 'pool'))
        # Hidden Layer
        for layer in range(n_layer-2):
            self.gcnlayer.append(SAGEConv(hid_feats, hid_feats, 'pool'))
            self.batchnorms.append(nn.BatchNorm1d(hid_feats))
        # Output Layer (output features for each node)
        self.gcnlayer.append(SAGEConv(hid_feats, out_feats, 'pool'))
        self.classify = nn.Linear(out_feats, n_classes)

    def forward(self, graph, node_feat, eweight=None):
        h = node_feat
        for i, layer in enumerate(self.gcnlayer):
            if(i == 0):
                h = F.leaky_relu(layer(graph, h))
            elif (i > 0) & (i < self.n_layer-1):
                h = F.leaky_relu(self.batchnorms[i-1](layer(graph, h)))
            else:
                h = layer(graph, h)
        
        graph.ndata['h'] = h  # Store node-level features
        
        # Return node-level predictions instead of graph-level
        return self.classify(h)