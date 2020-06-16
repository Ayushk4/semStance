# Input:
    # - att_weights and lstm_output from LSTMModel.forward()
    # - Data: 
        # - root_masks;
        # - child_masks;
        # - edge_labels (torch.long);
        # - child_to_root_labels (for h_child)


# Three different tensors:
    # - Root Nodes
    # - Child Nodes
    # - Edge Nodes

# Requires
    # - Child mask and root mask over LSTM for child nodes and root nodes vectors
    # - Edge embeddings layer corresponding to labels

# From Attended Rep; We get root nodes and child as a separate tensors by masks

# Stack Multiple following Stack units
    # In a single sem_graph unit: Input - Root Vectors, Child Vectors | Output - Root Vectors, Child Vectors
        # - h_root_i = MLP1([g_root; g_edge; g_child_i])
        # g'root = Attended over h_root_i for all child nodes i
        # g'child = h_child = MLP2([g_root; g_edge; g_child])
        # g''head = Linear(g'head) and g''child = Linear(g'child)
        # output Root and child Vectors = residual connection over (g_root, g''root) and (g_child, g''child) 

