import torch
from torch import nn
import numpy as np
import math
from params import params
from graph_att import graph_block

class LSTMModel(nn.Module):
    def __init__(self, embedding, embed_dims, num_edge_labels, num_graph_block, num_heads=10, graph_dropout=0.5,
                      classifier_mlp_hidden=16, bidirectional=True, dropout=0.1):
        super(LSTMModel, self).__init__()
        torch.manual_seed(params.torch_seed)

        self.model_type = "Embed_LSTM_ATT_MLP"
        self.padding_idx = len(embedding) - 1

        self.embed_dims = embed_dims
        if not params.concat:
            raise Exception("Bad idea! for params.concat")
        self.lstm_input_dims = embed_dims+2

        self.bidirection = bidirectional
        if not bidirectional:
            raise Exception("Unidirectional LSTM not supported")
        self.lstm_output_dims = self.lstm_input_dims * 2

        self.node_feat_dims = self.embed_dims
        self.dropout_mlp = nn.Dropout(p=dropout)
        self.classifier_mlp = nn.Sequential(nn.Linear(self.node_feat_dims, classifier_mlp_hidden),
                                    nn.Tanh(), self.dropout_mlp,
                                    nn.Linear(classifier_mlp_hidden, 4),
                                ) # 4 labels = Support, Refute, unrelated, comment
        self.lstm = nn.LSTM(embed_dims+2, embed_dims + 2, bidirectional=True)
        self.post_lstm_linear = nn.Linear(self.lstm_output_dims, self.node_feat_dims)

        self.att_score_linear = nn.Linear(self.node_feat_dims, 1)

        self.__init_weights__()
        self.num_edge_labels = num_edge_labels
        self.edge_label_embed = nn.Embedding(num_edge_labels, self.node_feat_dims)
        torch.nn.init.uniform_(self.edge_label_embed.weight, -0.1, 0.1)

        if num_graph_block > 0:
            self.graph_modules = nn.ModuleList([graph_block(self.node_feat_dims, num_heads, dropout=graph_dropout) for i in range(num_graph_block)])
        else:
            self.graph_modules = [lambda x,y,z: x]

        self.hidden = (torch.autograd.Variable(torch.zeros(2, 1, embed_dims+2)).to(params.device),   
                        torch.autograd.Variable(torch.zeros(2, 1, embed_dims+2)).to(params.device))

        self.embedding_layer = nn.Embedding(len(embedding), embed_dims, padding_idx=self.padding_idx)
        self.embedding_layer.weight.data.copy_(torch.Tensor(embedding))

        self.pos_encoder = PositionalEncoding(embed_dims, dropout)

    def __init_weights__(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, texts, target_buyer_vector, pad_masks, edge_indices, edge_labels, edge_masks):
        texts = self.embedding_layer(texts) * math.sqrt(self.embed_dims)
        texts = self.pos_encoder(texts)
        src_in = torch.cat((texts, target_buyer_vector), axis=2)

        # if self.input_dims != self.embed_dims + 2:
        #     src_in = self.embed2input_space(src_in)
        src_in = src_in.permute(1, 0, 2)

        h, c = (self.hidden[0].expand(-1, src_in.shape[1], -1).contiguous(),
                self.hidden[0].expand(-1, src_in.shape[1], -1).contiguous())
        l_out, _ = self.lstm(src_in, (h ,c))
        lstm_output =l_out.permute(1, 0, 2)

        graph_input = self.post_lstm_linear(lstm_output)
        edge_attr = self.edge_label_embed(edge_labels)

        nodes_attr = graph_input.reshape(-1, self.node_feat_dims)
        #print(nodes_attr.shape, edge_attr.shape, edge_masks.shape, edge_indices.shape)
        for module in self.graph_modules:
            #print(nodes_attr.shape, edge_attr.shape, edge_masks.shape, edge_indices.shape)
            nodes_attr = module(nodes_attr, edge_indices, edge_attr, edge_masks)
        nodes_attr = nodes_attr.view(graph_input.size(0), -1, self.node_feat_dims) + graph_input

        att_scores = self.att_score_linear(nodes_attr)
        att_scores = att_scores.masked_fill(pad_masks.unsqueeze(-1), -10000.0).softmax(1)
        weight_vector = torch.sum(nodes_attr * att_scores, 1)

        scores = self.classifier_mlp(weight_vector)
        return scores

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(1, 0, 2)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size()[0], :]
        return self.dropout(x)

if __name__ == "__main__":
    from dataloader import wtwtDataset
    import json
    dataset = wtwtDataset()
    train_dataset = dataset.train_dataset

    def open_it(pth):
        fo = open(pth, "r")
        j = json.load(fo)
        fo.close()
        return j
    embedding = open_it(params.glove_embed)

    model = LSTMModel(embedding, 200, 200, 3, graph_dropout=0, dropout=0.0, num_heads=10)
    model = model.to(params.device)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = []
    i = 0
    with torch.no_grad():
        for single_batch in train_dataset:
            # single_batch = train_dataset[0]
            (texts, stances, pad_masks, target_buyr, edge_indices, edge_labels, _, edge_masks) = single_batch
            scores = model(texts, target_buyr, pad_masks, edge_indices, edge_labels, edge_masks)
            loss.append(criterion(scores, stances).to("cpu").item())
            i += 1
            if i%100 == 0:
                print(i)
    print(sum(loss)/len(loss))
    # Karpathy test for avg = -ln(1/4)

    params = model.parameters()
    opt = torch.optim.Adam(params, lr=0.0001)

    model.train()
    single_batch = train_dataset[0]
    (texts, stances, pad_masks, target_buyr, edge_indices, edge_labels, _, edge_masks) = single_batch
    for i in range(10001):
        scores = model(texts, target_buyr, pad_masks, edge_indices, edge_labels, edge_masks)
        loss = criterion(scores, stances)
        loss.backward()
        opt.step()
        opt.zero_grad()

        if i % 100 == 0:
            print("%.6f" % loss.item())
    # Make sure overfit on single batch
