import torch
from torch import nn
import numpy as np
import math
from params import params
from graph_att import graph_att_submodel

class LSTMModel(nn.Module):
    def __init__(self, embedding, embed_dims, fusion_alpha, num_edge_labels, message_passing_hidden, 
            num_gatt_layers, gatt_dropout=0.7, classifier_mlp_hidden=16, bidirectional=True, dropout=0.1):
        super(LSTMModel, self).__init__()
        torch.manual_seed(params.torch_seed)

        self.fusion_alpha = fusion_alpha
        assert fusion_alpha >= 0 and fusion_alpha < 1
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

        self.last_att_linear = nn.Linear(self.lstm_output_dims, self.lstm_output_dims)
        self.last_att_tanh = nn.Tanh()

        self.sem_vector_dropout = nn.Dropout(p=dropout)
        self.sem_vector_linear = nn.Linear(self.lstm_output_dims, self.lstm_output_dims)

        self.dropout_mlp = nn.Dropout(p=dropout)
        self.classifier_mlp = nn.ModuleList([nn.Linear(self.lstm_output_dims, classifier_mlp_hidden),
                                    nn.Tanh(), self.dropout_mlp,
                                    nn.Linear(classifier_mlp_hidden, 4),
                                ]) # 4 labels = Support, Refute, unrelated, comment
        self.lstm = nn.LSTM(embed_dims+2, embed_dims+2, bidirectional=True)

        self.__init_weights__()####

        self.semantics_gatt_model = graph_att_submodel(num_edge_labels, self.lstm_output_dims, num_gatt_layers,
                                            message_passing_hidden, gatt_dropout)

        self.hidden = (torch.autograd.Variable(torch.zeros(2, 1, embed_dims+2)).to(params.device),   
                        torch.autograd.Variable(torch.zeros(2, 1, embed_dims+2)).to(params.device))

        self.embedding_layer = nn.Embedding(len(embedding), embed_dims, padding_idx=self.padding_idx)
        self.embedding_layer.weight.data.copy_(torch.Tensor(embedding))

        self.pos_encoder = PositionalEncoding(embed_dims, dropout)

    def __init_weights__(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, texts, target_buyer_vector, pad_masks, edge_labels, lstm_root_masks,
                    lstm_child_masks, gatt_masks_for_root, gatt_root_idxs, semantics_root_mask):
        texts = self.embedding_layer(texts) * math.sqrt(self.embed_dims)
        texts = self.pos_encoder(texts)
        src_in = torch.cat((texts, target_buyer_vector), axis=2)

        # if self.input_dims != self.embed_dims + 2:
        #     src_in = self.embed2input_space(src_in)
        src_in = src_in.permute(1, 0, 2)

        h, c = (self.hidden[0].expand(-1, src_in.shape[1], -1).contiguous(),
                self.hidden[0].expand(-1, src_in.shape[1], -1).contiguous())
        l_out, _ = self.lstm(src_in, (h ,c))
        lstm_output = l_out.permute(1, 0, 2)

        e_att = self.last_att_tanh(self.last_att_linear(lstm_output))
        att_weights = torch.sum(e_att * lstm_output, axis=2).masked_fill(pad_masks, -10000.0)

        semantics_vector = self.semantics_gatt_model(edge_labels, lstm_output, att_weights, lstm_root_masks, 
                                                lstm_child_masks, gatt_masks_for_root, gatt_root_idxs, semantics_root_mask)
        semantics_vector = self.sem_vector_dropout(self.sem_vector_linear(semantics_vector))
        # print(semantics_vector.size())

        att_weights = torch.softmax(att_weights, 1).unsqueeze(2)
        scores = torch.sum(att_weights * lstm_output, axis = 1)
        # print(scores.size())

        scores = (semantics_vector *self.fusion_alpha) + (scores * (1 - self.fusion_alpha))
        for module in self.classifier_mlp:
            scores = module(scores)

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

    model = LSTMModel(embedding, 200, 0.5, dataset.num_edge_labels, 404, 2, gatt_dropout=0.0,
                    classifier_mlp_hidden=16, bidirectional=True, dropout=0.0)
    model = model.to(params.device)

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    loss = []
    i = 0
    # with torch.no_grad():
    #     for single_batch in train_dataset:
    #         # single_batch = train_dataset[0]

    #         (texts, stances, pad_masks, target_buyr, edge_labels, lstm_root_masks, lstm_child_masks,
    #             gatt_masks_for_root, gatt_root_idxs, semantics_root_mask) = single_batch

    #         scores = model(texts, target_buyr, pad_masks, edge_labels, lstm_root_masks,
    #                         lstm_child_masks, gatt_masks_for_root, gatt_root_idxs, semantics_root_mask)
    #         loss.append(criterion(scores, stances).to("cpu").item())

    #         i += 1
    #         if i%100 == 0:
    #             print(i)
    
    # print(sum(loss))
    # Karpathy test for avg = -ln(1/4)

    params = model.parameters()
    opt = torch.optim.SGD(params, lr=0.00001)

    model.train()
    single_batch = train_dataset[0]
    (texts, stances, pad_masks, target_buyr, edge_labels, lstm_root_masks, lstm_child_masks,
            gatt_masks_for_root, gatt_root_idxs, semantics_root_mask) = single_batch
    for i in range(1000):

        scores = model(texts, target_buyr, pad_masks, edge_labels, lstm_root_masks,
                            lstm_child_masks, gatt_masks_for_root, gatt_root_idxs, semantics_root_mask)
        loss = criterion(scores, stances)
        loss.backward()
        opt.step()

        print(loss)        
    # Make sure overfit on single batch    

