# CHECK ~ of bool tensor
# Input:
    # - att_weights and lstm_output from LSTMModel.forward()
    # - Data: 
        # - root_masks;
        # - child_masks;
        # - edge_labels (torch.long);
        # - child_to_root_labels (for h_child)
        # - root to child masks

# Three different tensors:
    # - Root Nodes
    # - Child Nodes
    # - Edge Nodes

# Requires
    # - Child mask and root mask over LSTM for child nodes and root nodes vectors
    # - Edge embeddings layer corresponding to labels

# From Attended Rep; We get root nodes and child as a separate tensors by masks
# We get edge vectors from learnable edge embedding

# Stack Multiple following Stack units
    # In a single sem_graph unit: Input - Root Vectors, Child Vectors | Output - Root Vectors, Child Vectors
        # - h_root_i = MLP1([g_root; g_edge; g_child_i])
        # g'root = Attended over h_root_i for all child nodes i
        # g'child = h_child = MLP2([g_root; g_edge; g_child])
        # g''head = Linear(g'head) and g''child = Linear(g'child)
        # output Root and child Vectors = residual connection over (g_root, g''root) and (g_child, g''child) 

import torch
from torch import nn
import dataloader
from params import params


class graph_att_submodel(nn.Module):
    def __init__(self, num_edge_labels, lstm_out_dims, num_gatt_layers, message_passing_hidden, gatt_dropout):
        super(graph_att_submodel, self).__init__()
        assert message_passing_hidden > 0
        assert num_gatt_layers < 10
        self.edge_label_pad_idx = num_edge_labels - 1

        self.embedding_layer = nn.Embedding(num_edge_labels, lstm_out_dims, padding_idx=self.edge_label_pad_idx)
        self.__init_weights__()

        self.gatt_layer = nn.ModuleList([single_gatt_layer(lstm_out_dims, message_passing_hidden, gatt_dropout)
                                        for i in range(num_gatt_layers)])

        self.final_root_att = nn.Linear(lstm_out_dims, lstm_out_dims)

    def __init_weights__(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if name[-4:] == "bias":
                    nn.init.zeros_(p)
                else:
                    nn.init.xavier_uniform_(p)

    def forward(self, edge_labels, lstm_out, att_weights, lstm_root_masks, lstm_child_masks,
                gatt_masks_for_root, gatt_root_idxs, semantics_root_mask):

        edge_vectors = self.embedding_layer(edge_labels)

        root_att_wts = att_weights.unsqueeze(1).expand(-1, dataloader.ROOT_NODES_PADDED_LEN, -1)
        root_att_wts = torch.softmax(root_att_wts.masked_fill(lstm_root_masks, -10000.0), 2).unsqueeze(3)
        root_vectors = torch.sum(root_att_wts * \
                                 (lstm_out.unsqueeze(1).expand(-1, dataloader.ROOT_NODES_PADDED_LEN, -1, -1)),
                                axis = 2)

        child_att_wts = att_weights.unsqueeze(1).expand(-1, dataloader.CHILD_NODES_PADDED_LEN, -1)
        child_att_wts = torch.softmax(child_att_wts.masked_fill(lstm_child_masks, -10000.0), 2).unsqueeze(3)
        child_vectors = torch.sum(child_att_wts * \
                                  (lstm_out.unsqueeze(1).expand(-1, dataloader.CHILD_NODES_PADDED_LEN, -1, -1)),
                                axis = 2)
        
        for layer in self.gatt_layer:
            root_vectors, child_vectors = layer(root_vectors, edge_vectors, child_vectors,
                                                gatt_masks_for_root, gatt_root_idxs)

        final_att = torch.tanh(self.final_root_att(root_vectors))
        final_att_weights = torch.sum(final_att * root_vectors, axis=2).masked_fill(semantics_root_mask, -10000.0)
        final_att_weights = torch.softmax(final_att_weights, 1).unsqueeze(2)

        semantic_vector = torch.sum(final_att_weights * root_vectors, axis=1)
        return semantic_vector

class single_gatt_layer(nn.Module):
    def __init__(self, vector_dims, message_passing_hidden, gatt_dropout, same_root_child_mlp=False):
        super(single_gatt_layer, self).__init__()
        
        if same_root_child_mlp == True:
            print("same_root_child_mlp == True NOT supported yet.")

        self.vector_dims = vector_dims
        self.message_passing_hidden = message_passing_hidden

        self.dropout_mlp_root = nn.Dropout(p=gatt_dropout)
        self.mlp_root = nn.ModuleList([nn.Linear(3*self.vector_dims, self.message_passing_hidden),
                                    nn.Tanh(), self.dropout_mlp_root,
                                    nn.Linear(self.message_passing_hidden, self.vector_dims),
                                ])

        self.message_pass_att = nn.Linear(2*self.vector_dims, self.vector_dims)

        self.dropout_mlp_child = nn.Dropout(p=gatt_dropout)
        self.mlp_child = nn.ModuleList([nn.Linear(3*self.vector_dims, self.message_passing_hidden),
                                    nn.Tanh(), self.dropout_mlp_child,
                                    nn.Linear(self.message_passing_hidden, self.vector_dims),
                                ])

        self.dropout_final_linear = nn.Dropout(p=gatt_dropout)
        self.final_linear = nn.Linear(self.vector_dims, self.vector_dims)
        self.__init_weights__()

    def __init_weights__(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if name[-4:] == "bias":
                    nn.init.zeros_(p)
                else:
                    nn.init.xavier_uniform_(p)

    def forward(self, root_vecs, edge_vecs, child_vecs, gatt_masks_for_root, gatt_root_idxs):
        # root = MLP1([g_root; g_edge; g_child]) ; is vector concatenation.
        #     after this we obtain g'_root by attention: softmax(matmul(h_root, linear(h_root;g_root))).
        # g'child = h_child = MLP2([g_root; g_edge; g_child])
        # g''root = Linear(g'head) and g''child = Linear(g'child)
        # Followed by the Residual Connection on root and child vectors 

        root_expanded = root_vecs.unsqueeze(2).expand(-1, -1, dataloader.CHILD_NODES_PADDED_LEN, -1)
        root_childwise = torch.cat([torch.index_select(tensr, 0, idx).unsqueeze(0) for tensr, idx in zip(root_vecs, gatt_root_idxs)],axis=0)
        child_edge_cat = torch.cat((child_vecs, edge_vecs), axis=2)
        child_edge_expanded = child_edge_cat.unsqueeze(1).expand(-1, dataloader.ROOT_NODES_PADDED_LEN, -1, -1)

        h_root = torch.cat((root_expanded, child_edge_expanded), axis=3)
        for module in self.mlp_root:
            h_root = module(h_root)

        e_att = self.message_pass_att(torch.cat((h_root, root_expanded), axis=3))
        att_weights = torch.softmax(torch.sum(e_att * h_root, axis=3).masked_fill(gatt_masks_for_root, -10000.0), 2).unsqueeze(3)
        g_root = torch.sum(att_weights * h_root, axis = 2)

        g_child = torch.cat((root_childwise, child_edge_cat), axis=2)
        for module in self.mlp_child:
            g_child = module(g_child)
        assert g_child.size() == child_vecs.size()
        
        g_child = child_vecs + self.dropout_final_linear(self.final_linear(g_child))
        g_root = root_vecs + self.dropout_final_linear(self.final_linear(g_root))

        return g_root, g_child
        
        

if __name__ == "__main__":
    layer_instance = single_gatt_layer(204, 204, gatt_dropout=0.8)
    root_vecs = torch.randn(16, dataloader.ROOT_NODES_PADDED_LEN, 204)
    edge_vecs = torch.randn(16, dataloader.CHILD_NODES_PADDED_LEN, 204)
    child_vecs = torch.randn(16, dataloader.CHILD_NODES_PADDED_LEN, 204)

    dataset = dataloader.wtwtDataset()
    train_dataset = dataset.train_dataset

    single_batch = train_dataset[0]
    (texts, stances, pad_masks, target_buyr, edge_labels,
    lstm_root_masks, lstm_child_masks, gatt_masks_for_root,
    gatt_root_idxs, semantics_root_mask) = single_batch

    layer_instance(root_vecs, edge_vecs, child_vecs, gatt_masks_for_root, gatt_root_idxs)
