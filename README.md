# SemStance

Can semantics help with stance detection.
The Dataset used is Will-they-wont-they dataset (ACL'2020).

## Directory Layout

- README.md
- src
- utils
- data


- First prepare data by running pre_process.py (for tokenizing) followed by normalize.py (for normalizing target, buyer... etc.)
- Then sdp/cnvrt2_sdp_format.py for convert to parsable format by sdp, then sdp/run_am_parser.py to parse, then sdp/sdp_2_edgelist_dict.py to collate together, then sdp/unique_edges.py for finding unique_edges and indexing tp get `edges_types.json` and `edges_idxed.json, then sdp/prep_sdp_dataset.py for merging normalized.json with edges_idxed.json into `sdp/prepped_sdp.json`
- Then prepare glove by running smaller_glove.py (prepares smaller glove based on occurrences in tokens) and prepare_glove.py (add to vocab from smaller glove withmore than 10 occurence if not present)
- Then Index by index_dataset.py

- Then start training the model


# To Try

- Try Edge features as embedding vectors and as random vectors
- Separate MLP for Root and Child nodes
- Normalize based on number of root nodes while doing the fusion
