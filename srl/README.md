1. First run `graph_preprocess.py` to pre_process raw tweets into pre_processed.json. This works similarly to ../data/pre_process.json, except some minor changes to work better on SRL model - No lowercasing, maps <number>, <user> and <money> to 0, $1, Some Proper Noun for SRL to run. Also 's and \n\n to some key words to help split sentences.

    Output from above => "./data/pre_processed.json"

2. Next run `graph_sent_split.py` to split each tweet into sentence about [?, !, :, ;] and non-breaking "." Also prepare input text for LSTM example: re.sub(" sasasasasas", "'s", period_splits)

    Output from above => "./data/splitted.json"

3. Next run `graph_create.py <start> <end>` where `<start>` and `<end>` are staring and ending indices with regards to dataset to run off-the-shelf semantic graph parser.

    Output from above => "./data/srl<start>_<end>.json" 

    Run above Till it covers entire dataset

4. Next run `graph_normalize_text.py` to normalize the text (tokens) View the comments in the file for more details on how it works.

    Output from above => "./data/normalized.json" 

5. Next run `graph_mapping.py` to create graph mapping from tokens to nodes (root and child), edge nodes, edge_indices.

    Output from above => "./data/mapped.json"

6. Lastly run `graph_batching.py` to make the graph ready for batching (padding node vectors ... etc.).

    Output from above => "./data/batching.json"
