import argparse

parser = argparse.ArgumentParser()

dataset_path = "sdp/indexed.json"
glove_embed = "glove/embed_glove.json"
glove_dims_ = 200
num_edge_labels = 26

parser.add_argument("--python_seed", type=int, default=49)
parser.add_argument("--torch_seed", type=int, default=4214)
parser.add_argument("--target_merger", type=str, default="Please Enter Test Merger in args", help="Test Merger in 'CVS_AET', 'ANTM_CI', 'AET_HUM', 'CI_ESRX'")
parser.add_argument("--test_mode", type=str, default="True")
parser.add_argument("--cross_valid_num", type=int, default=4, help="For 5-fold crossvalidation, which part is valid set.")

parser.add_argument("--dataset_path", type=str, default=dataset_path)
parser.add_argument("--glove_embed", type=str, default=glove_embed)

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--n_epochs", type=int, default=100)
 
parser.add_argument("--num_graph_blocks", type=int, default=3, help="Number of graph layers stacked up.")
parser.add_argument("--graph_dropout", type=float, default=0.4, help="Dropout for graph model and its layers.")

parser.add_argument("--dropout", type=float, default=0.5, help="Dropout for position encoder and MLP classifier.")
parser.add_argument("--mlp_hidden", type=int, default=16, help="Hidden dims size for a 2 layer MLP used for bringing attention lstm to tag space")
parser.add_argument("--glove_dims", type=int, default=glove_dims_, help="Dimensions of glove twitter embeddings.")
# parser.add_argument("--lstm_input_dims", type=int, default=102, help="Input_dimensions for LSTM.")

parser.add_argument("--dummy_run", dest="dummy_run", action="store_true", help="To make the model run on only one training sample for debugging")
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")
parser.add_argument("--concat", type=bool, default=True, help="Whether [0, 1] for target and [1, 0] should be concatenated or added")

parser.add_argument("--run", type=str, default=None)
parser.add_argument("--wandb",  dest="wandb", action="store_true", default=False)

params = parser.parse_args()

assert params.target_merger in ['CVS_AET', 'ANTM_CI', 'AET_HUM', 'CI_ESRX']
assert params.cross_valid_num >= 0 and params.cross_valid_num <=4
