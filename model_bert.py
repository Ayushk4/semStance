import torch
from torch import nn
import numpy as np
import math
from params import params
from graph_att import graph_block
from transformers import BertModel

class BertStanceModel(nn.Module):
    def __init__(self, num_edge_labels, num_graph_block, graph_dropout=0.5, classifier_mlp_hidden=16):
        super(BertStanceModel, self).__init__()
        torch.manual_seed(params.torch_seed)

        self.model_type = "BERT"
        self.bert = BertModel.from_pretrained(params.bert_type)

        self.node_feat_dims = self.bert.config.hidden_size
        self.att_score_linear = nn.Linear(self.node_feat_dims, 1)
        self.classifier_mlp = nn.Sequential(nn.Linear(self.node_feat_dims, classifier_mlp_hidden),
                                    nn.Tanh(), nn.Dropout(p=self.bert.config.hidden_dropout_prob),
                                    nn.Linear(classifier_mlp_hidden, 4),
                                ) # 4 labels = Support, Refute, unrelated, comment

        self.num_edge_labels = num_edge_labels
        self.edge_label_embed = nn.Embedding(num_edge_labels, self.node_feat_dims)
        torch.nn.init.uniform_(self.edge_label_embed.weight, -0.1, 0.1)

        if num_graph_block > 0:
            self.graph_modules = nn.ModuleList([graph_block(self.node_feat_dims, graph_dropout) for i in range(num_graph_block)])
        else:
            self.graph_modules = [lambda x,y,z,w: x]

        self.new_special_tokens_dict = {"additional_special_tokens": ["<number>", "<money>", "<user>"]}
        self.add_special_tokens(len(self.new_special_tokens_dict["additional_special_tokens"]))
    
    def add_special_tokens(self, num_tokens):
        print("Embeddings type:", self.bert.embeddings.word_embeddings.weight.data.type(),
                "Embeddings shape:", self.bert.embeddings.word_embeddings.weight.data.size())
        embedding_size = self.bert.embeddings.word_embeddings.weight.size(1)
        new_embeddings = torch.FloatTensor(num_tokens, embedding_size).uniform_(-0.1, 0.1)
        print("new_embeddings shape:", new_embeddings.size())
        new_embedding_weight = torch.cat((self.bert.embeddings.word_embeddings.weight.data,new_embeddings), 0)
        self.bert.embeddings.word_embeddings.weight.data = new_embedding_weight
        print("Embeddings shape:", self.bert.embeddings.word_embeddings.weight.data.size())

    def forward(self, texts, target_buyer_vector, pad_masks, edge_indices, edge_labels, edge_masks):
        assert target_buyer_vector == None and pad_masks == None
        pad_masks = (texts != 0) * 1
        outputs = self.bert(texts, attention_mask=pad_masks)

        graph_input = outputs[0]

        edge_attr = self.edge_label_embed(edge_labels)
        nodes_attr = graph_input.reshape(-1, self.node_feat_dims)
        for module in self.graph_modules:
            nodes_attr = module(nodes_attr, edge_indices, edge_attr, edge_masks)
        nodes_attr = nodes_attr.view(graph_input.size(0), -1, self.node_feat_dims) + graph_input

        att_scores = self.att_score_linear(nodes_attr)
        att_scores = att_scores.masked_fill(pad_masks.unsqueeze(-1), -10000.0).softmax(1)
        weight_vector = torch.sum(nodes_attr * att_scores, 1)

        scores = self.classifier_mlp(weight_vector)
        return scores

if __name__ == "__main__":
    tmp = torch.randn(5,5).cuda()
    from bertloader import wtwtDataset
    import json
    dataset = wtwtDataset()
    train_dataset = dataset.train_dataset
    model = BertStanceModel(200, 3, graph_dropout=0.0, classifier_mlp_hidden=16)
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
    opt = torch.optim.Adam(params, lr=0.00001)

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

