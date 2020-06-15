First run `graph_preprocess.py` to pre_process raw tweets into pre_processed.json. This works similarly to ../data/pre_process.json, except some minor changes to work better on SRL model - No lowercasing, maps <number>, <user> and <money> to 0, $1, Some Proper Noun for SRL to run. Also 's and \n\n to some key words to help split sentences.

Output from above => "./data/pre_processed.json"

Next run `graph_sent_split.py` to split each tweet into sentence about [?, !, :, ;] and non-breaking "." Also prepare input text for LSTM example: re.sub(" sasasasasas", "'s", period_splits)


Output from above => "./data/splitted.json"

# TODO:
[\'\’] instead of \'  in data/preprocess.py
