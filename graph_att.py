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
import dataloader

class graph_att(nn.Module):
    def __init__(self, num_edge_labels, lstm_out_dims, num_gatt_layers, message_passing_hidden, gatt_dropout):
        super(graph_att, self).__init__()
        self.edge_label_pad_idx = num_edge_labels - 1

        self.embedding_layer = nn.Embedding(num_edge_labels, lstm_out_dims, padding_idx=self.edge_label_pad_idx)

        __init_weights__()

    def __init_weights__(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, att_weights, lstm_root_masks, lstm_child_masks, root_idxs_for_child,
                child_masks_for_root, edge_labels, semantics_root_mask):
        pass

class single_gatt_layer(nn.Module):
    def __init__(self, vector_dims, message_passing_hidden, gatt_dropout, same_root_child_mlp=True):
        super(single_graph_layer, self).__init__()
        
        if same_root_child_mlp == False:
            print("same_root_child_mlp == False NOT supported yet.")

        self.vector_dims = vector_dims
        self.message_passing_hidden = message_passing_hidden

        self.dropout_mlp_root = nn.Dropout(p=gatt_dropout)
        self.mlp_root = nn.ModuleList([nn.Linear(3*self.vector_dims, self.message_passing_hidden),
                                    nn.Tanh(), self.dropout_mlp_root,
                                    nn.Linear(self.message_passing_hidden, self.vector_dims),
                                ])

        self.message_pass_att = nn.Linear(2*self.vector_dims, self.vector_dims)

        # self.dropout_mlp_child = nn.Dropout(p=gatt_dropout)
        # self.mlp_child = nn.ModuleList([nn.Linear(3*f.vector_dims, self.message_passing_hidden),
        #                             nn.Tanh(), self.dropout_mlp_child,
        #                             nn.Linear(self.message_passing_hidden, self.vector_dims),
        #                         ])
        
        self.dropout_final_linear = nn.Dropout(p=gatt_dropout)
        self.final_linear = nn.Linear()
        __init_weights__()

    def __init_weights__(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, root_vecs, edge_vecs, child_vecs, root_idxs_for_child, child_masks_for_root):
        # root = MLP1([g_root; g_edge; g_child]) ; is vector concatenation.
        #     after this we obtain g'_root by attention: softmax(matmul(h_root, linear(h_root;g_root))).
        # g'child = h_child = MLP2([g_root; g_edge; g_child])
        # g''root = Linear(g'head) and g''child = Linear(g'child)
        # Followed by the Residual Connection on root and child vectors 

        g_root_expanded = root_vecs.unsqueeze(2).expand(-1, -1, dataloader.CHILD_NODES_PADDED_LEN, -1)
        h_root = torch.cat((g_root_expanded,
                            torch.cat((child_vecs, edge_vecs), axis=2).unsqueeze(1).expand(-1, dataloader.ROOT_NODES_PADDED_LEN, -1, -1)
                            ), 
                            axis=3
        )

        for module in self.classifier_mlp_root:
            h_root = module(h_root)
        
        h_child == h_root.clone()
        assert h_child.size() == torch.Size([root_vecs.size(0), root_vecs.size(1), child_vecs.size(1), root_vecs.size(2)])

if __name__ == "__main__":
    layer = single_gatt_layer(204, 204, gatt_dropout=0.8)
    root_vecs = torch.randn(16, dataloader.ROOT_NODES_PADDED_LEN, 204)
    edge_vecs = torch.randn(16, dataloader.CHILD_NODES_PADDED_LEN, 204)
    child_vecs = torch.randn(16, dataloader.CHILD_NODES_PADDED_LEN, 204)
    
    
    
