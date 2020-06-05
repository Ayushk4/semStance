# SemStance

Can semantics help with stance detection.
The Dataset used is Will-they-wont-they dataset (ACL'2020).

## Directory Layout

- README.md
- src
- utils
- data


- First prepare data by running pre_process.py (for tokenizing) followed by normalize.py (for normalizing target, buyer... etc.)
- Then prepare glove by running smaller_glove.py (prepares smaller glove based on occurrences in tokens) and prepare_glove.py (add to vocab from smaller glove withmore than 10 occurence if not present)
- Then Index by index_dataset.py

- Then start training the model


# To Try

- Transformer xavier init
- Transformer use CLS for classification
- Layer Norm in practise
- First project embedding, then add position_embedding
- Extra vectors [1, 0] expand due to dropout
- FFN in transformer set dims = 4 * transformer_ip_dims
- Label smoothing on classification

