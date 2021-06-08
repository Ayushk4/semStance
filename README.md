# SemStance

Stance Detection hold a lot of applications for analysis in online media and in rumour verification/fact-checking. This projects aimed at imrpoving the stance detection systems by incorporating semantic graph structure into the model using grpah attention mechanisms. We obtained massive performance gains over the baselines from WT-WT paper. This branch contains the code for multi-headed graph attention over semantic dependency graphs to improve stance detection. Please check the bert branch for the same over Bert. 

## Getting Started

First clone this repository, then:

### Setup the dataset

- Download the labelled dataset from [this link](https://github.com/cambridge-wtwt/acl2020-wtwt-tweets/blob/master/wtwt_ids.json).

- Please obtain the wtwt dataset annotation (tweetid-stance-labels) file from [this GitHub repository](https://github.com/cambridge-wtwt/acl2020-wtwt-tweets) and save it with the filename `data/wtwt_ids.json`

- To extract the tweets content please register your application on the twitter developer API and download the tweets. Save all the tweets in a single folder named `data/scrapped_full/` with each file named in the format `<tweet_id>.json` where tweet_id is a 17-20 digit tweet id. Add the desired target sentences for each merger in merger2target.json inside this folder.

- To prepare the dataset, please set up the dependencies and follow the above two steps. Then execute - `cd data`; `python3 pre_process.py` and then `python3 normalize.py`.


### Prerequisites

- `Pytorch > 1.0`
- `Transformers > 2.9`
- `Torch Geometric`
- `Wandb`
- `scikit-learn`
- `scipy`
- `ekphrasis`
- `numpy`

**SRL Graphs**: After obtaining the dataset, use the semantic parser from [coli-saar/am-parser](https://github.com/coli-saar/am-parser) to obtain the semantic dependency graphs and store them in `srl` folder.

**Glove Embeddings**: Obtain the glove embeddings for Twitter from https://nlp.stanford.edu/projects/glove/

### Experiments

The targets have same code as the WT-WT paper.

To train the model `train.py --tar <target_required>`.

For bert use `train_bert.py --tar <target_required>` for the code in `bert` branch.

Following additional hyperparameters may be used:

| Hyperparameters  | Value |
| ---------------  | ----- |
| learning rate    | {1e-4, 3e-4, 1e-5}  |
| concat           | true  |
| dropout          | 0.5   |
| n_epochs         | 50    |
| num_heads        | 10    |
| batch_size       | 16    |
| glove_dims       | 200   |
| graph_dropout    | 0.7   |
| num_graph_blocks | 3,5,7 |

## Experiments and Architectures

Code for various architectures can be found in their respective branches.

## Running the tests

Set the Testing Flag to True.

## Author

Ayush Kaushal - [ayushk4](https://github.com/Ayushk4)

## Misc

- Project was carried out under Niloy Ganguly, IIT Kharagpur
- License: MIT
- Code written from scratch by Ayush Kaushal (GitHub: ayushk4)

