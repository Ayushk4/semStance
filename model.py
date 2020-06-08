import torch
import torch.nn as nn
from params import params
import math

# Model Specification:
# src = self.encoder(src) * math.sqrt(self.ninp)
# src = self.pos_encoder(src)
# src = src.concat(src, target_buyer_vector)
# output = self.transformer_encoder(src, self.src_mask) - 12 layers of transformer encoder
# e_i = tanh(wh_i + b) || a = softmax(matmul(e,h_i) * pad_mask)
# o = sum(a_i*e_i) for i =  1...70
# output = MLP(o)

class TransformerModel(nn.Module):
    def __init__(self, embedding, embed_dims, trans_input_dims, num_heads, hidden_dims, num_layers, classifier_mlp_hidden=16, dropout=0.5):
        super(TransformerModel, self).__init__()
        torch.manual_seed(params.torch_seed)

        self.model_type = "Transformer"
        self.padding_idx = len(embedding) - 1

        if params.concat:
            self.final_embed_dims = embed_dims + 2
        else:
            raise Exception("Bad idea! for params.concat")

        self.input_dims = trans_input_dims
        if trans_input_dims != embed_dims + 2:
            self.embed2input_space = nn.Linear(self.final_embed_dims, self.input_dims)

        encoder_layers = nn.TransformerEncoderLayer(self.input_dims, num_heads, hidden_dims, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.transformer_output_dims = self.input_dims
        self.last_att_linear = nn.Linear(self.transformer_output_dims, self.transformer_output_dims)
        self.last_att_tanh = nn.Tanh()

        self.dropout_mlp = nn.Dropout(p=dropout)
        self.classifier_mlp = nn.ModuleList([nn.Linear(self.transformer_output_dims, classifier_mlp_hidden),
                                    nn.Tanh(), self.dropout_mlp,
                                    nn.Linear(classifier_mlp_hidden, 4),
                                    # nn.Softmax(dim=1)
                                ]) # 4 labels = Support, Refute, unrelated, comment

        self.__init_weights__()

        self.embed_dims = embed_dims
        self.embedding_layer = nn.Embedding(len(embedding), embed_dims, padding_idx=self.padding_idx)
        self.embedding_layer.weight.data.copy_(torch.Tensor(embedding))

        self.pos_encoder = PositionalEncoding(embed_dims, dropout)

    def __init_weights__(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, texts, target_buyer_vector, pad_masks):
        texts = self.embedding_layer(texts) * math.sqrt(self.embed_dims)
        texts = self.pos_encoder(texts)
        src_in = torch.cat((texts, target_buyer_vector), axis=2)

        if self.input_dims != self.embed_dims + 2:
            src_in = self.embed2input_space(src_in)
        src_in = src_in.permute(1, 0, 2)

        trans_output = self.transformer_encoder(src_in, src_key_padding_mask=pad_masks).permute(1, 0, 2)

        e_att = self.last_att_tanh(self.last_att_linear(trans_output))
        att_weights = torch.softmax(torch.sum(e_att * trans_output, axis=2).masked_fill(pad_masks, -10000.0), 1).unsqueeze(2)
        scores = torch.sum(att_weights * trans_output, axis = 1)

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
    import json
    from dataloader import wtwtDataset
    torch.manual_seed(params.torch_seed)
    def open_it(pth):
        fo = open(pth, "r")
        j = json.load(fo)
        fo.close()
        return j

    glove_embed = open_it("glove/embed_glove.json")
    
    # print(type(glove_embed) , len(glove_embed), glove_embed[0])
    print(torch.Tensor(glove_embed).size())

    dataset = wtwtDataset()
    print("\n\n")
    model = TransformerModel(glove_embed, embed_dims=100, trans_input_dims=102,
                                num_heads=3, hidden_dims=100+2, num_layers=3,
                                classifier_mlp_hidden=12, dropout=0.0)
    model = model.to(params.device)

    loss = []
    criterion = nn.CrossEntropyLoss(reduction='sum')
    o = torch.zeros(4).to(params.device)
    with torch.no_grad():
        for i in range(len(dataset.train_dataset)):
            texts, stances, pad_masks, target_buyr = dataset.train_dataset[i]
            preds = model(texts, target_buyr, pad_masks)
            loss.append(criterion(preds, stances))
            o += torch.sum(preds, axis=0)

    print(len(loss), loss[0].item())
    import numpy as np
    print(np.sum([i.item() for i in loss]))
    print(o)
