import argparse

parser = argparse.ArgumentParser()

dataset_path = "data/data/indexed.json"
glove_embed = "glove/embed_glove.json"
glove_dims_ = 200

parser.add_argument("--python_seed", type=int, default=49)
parser.add_argument("--torch_seed", type=int, default=4214)
parser.add_argument("--target_merger", type=str, default="Please Enter Test Merger in args", help="Test Merger in 'CVS_AET', 'ANTM_CI', 'AET_HUM', 'CI_ESRX'")
parser.add_argument("--test_mode", dest="test_mode", action="store_true", help="If non given then train on train+valid and eval on test, else train on train, eval on valid")

parser.add_argument("--dataset_path", type=str, default=dataset_path)
parser.add_argument("--glove_embed", type=str, default=glove_embed)

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--n_epochs", type=int, default=50)
# parser.add_argument("--patience", type=int, default=7)


parser.add_argument("--dropout", type=float, default=0.7, help="Dropout for position encoder and transformer layers")
parser.add_argument("--mlp_hidden", type=int, default=16, help="Hidden dims size for a 2 layer MLP used for bringing attention transformer to tag space")
parser.add_argument("--num_heads", type=int, default=8, help="Number of parallel attention heads in each transformer encoder layer.")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers stacks of transformer encoder.")
parser.add_argument("--trans_ff_hidden", type=int, default=glove_dims_, help="Transformer attention hidden dims")
parser.add_argument("--glove_dims", type=int, default=glove_dims_, help="Dimensions of glove twitter embeddings.")
parser.add_argument("--trans_ip_dims", type=int, default=512, help="Input_dimensions for transformer.")

parser.add_argument("--dummy_run", dest="dummy_run", action="store_true", help="To make the model run on only one training sample for debugging")
parser.add_argument("--device", type=str, default="cuda", help="name of the device to be used for training")
parser.add_argument("--concat", type=bool, default=True, help="Whether [0, 1] for target and [1, 0] should be concatenated or added")

parser.add_argument("--run", type=str, default=None)
parser.add_argument("--wandb",  dest="wandb", action="store_true", default=False)

params = parser.parse_args()

assert params.target_merger in ['CVS_AET', 'ANTM_CI', 'AET_HUM', 'CI_ESRX']
