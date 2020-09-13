import torch
from torch import nn
import torch_geometric
from torch_geometric import nn as geo_nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform

class graph_block(nn.Module):
    def __init__(self, feat_dims, dropout, use_norm=True):
        assert type(use_norm) == type(True) and type(dropout) == float
        super(graph_block, self).__init__()
        assert type(use_norm) == type(True)
        self.gcn = GraphConv(feat_dims, dropout=dropout)
        if use_norm:
            self.norm1 = nn.LayerNorm(feat_dims)
            self.norm2 = nn.LayerNorm(feat_dims)
        else:
            self.norm1 = self.norm2 = lambda x: x
        self.linear = nn.Linear(feat_dims, feat_dims)

    def forward(self, x, edge_index, edge_attr, edge_masks):
        #print(x.shape, edge_index.shape, edge_attr.shape)
        x = self.norm1(self.gcn(x, edge_index, edge_attr, edge_masks))
        x = self.norm2(x + self.linear(x))
        return x

class GraphConv(MessagePassing):
    def __init__(self, in_channels, dropout=0.5, aggr='add', root_weight=True, bias=True, mh_dropout=0.1, **kwargs):
        super(GraphConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = in_channels
        in_c = in_channels
        self.dropout = dropout
        self.att_nn = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_c*3, in_c, bias=True), nn.Tanh(),
                                    nn.Linear(in_c, 1, bias=False), nn.LeakyReLU(0.2))
        self.nn = nn.Linear(in_c, in_c)
        self.aggr = aggr
        #self.mh_att = nn.MultiheadAttention(in_channels, num_heads, dropout=mh_dropout)

        if root_weight:
            self.root = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_attr, edge_masks):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_masks=edge_masks)

    def message(self, x_i, x_j, edge_attr, edge_masks=None):
        w_x = [self.nn(x_i), self.nn(x_j), self.nn(edge_attr)]
        att_scores = self.att_nn(torch.cat(w_x, 1)).squeeze(1)
        att_scores_e = att_scores.expand(att_scores.shape[0], att_scores.shape[0])
        att_scores_softmaxed = att_scores_e.masked_fill(~edge_masks, -10000.0).softmax(1)
        # print(att_scores.shape, att_scores_e.shape, att_scores_softmaxed.shape, x_j.shape, att_scores_softmaxed.sum(), att_scores_softmaxed.min(), att_scores_softmaxed.max(), att_scores_softmaxed.diagonal())

        return w_x[1] * att_scores_softmaxed.diagonal().unsqueeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

if __name__ == "__main__":
    from dataloader import wtwtDataset
    from params import params
    import json
    dataset = wtwtDataset()
    train = dataset.train_dataset

    def open_it(pth):
        fo = open(pth, "r")
        j = json.load(fo)
        fo.close()
        return j
    embedding = open_it(params.glove_embed)
    padding_idx = len(embedding) - 1
    embedding_layer = nn.Embedding(len(embedding), 200, padding_idx=padding_idx)
    embedding_layer.weight.data.copy_(torch.Tensor(embedding))
    embedding_layer = embedding_layer.to("cuda")
    embedding_layer.weight.requires_grad = False

    g = graph_block(200, 10, 0, use_norm=False).to("cuda")

    data = datapoint = train[0]
    text = data[0]
    edge_indices = data[4]
    edge_attr = torch.randn(edge_indices.size(1), 200).to('cuda')
    edge_masks = data[7]
    #print(edge_attr.shape, edge_indices.shape)

    embed = embedding_layer(text)
    embed = embed.view(-1, embed.shape[2])
    #print(edge_indices[0, :])
    #print(embed.shape)
    x = g(embed, edge_indices, edge_attr, edge_masks)
    print("\n\n=============\n\n", x.shape)

    criterion = torch.nn.MSELoss(reduction='sum')
    params = g.parameters()
    opt = torch.optim.Adam(params, lr=0.0004)

    print(data) 

    g.train()
    for i in range(10000):
        scores = g(embed, edge_indices, edge_attr, edge_masks)
        loss = scores.abs().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        if i % 5 == 0:
            print(loss.item())
