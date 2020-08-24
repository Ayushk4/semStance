# I/P: data/pre_processed.json glove/glove.6B/glove.6B.100d.txt
# O/P: data/padded_indexed.json

# Read Glove and pre_processed files
# For each tweet:
#   if occurs more than 3 times
#     if in glove then add to vocab, else if more than 20 then 
#   

import argparse
import json
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--glove_dims", default=200, type=int)
params = parser.parse_args()

GLOVE_PATH = "glove/glove.twitter.27B." + str(params.glove_dims) + "d.txt"
print(GLOVE_PATH)
DATASET = "sdp/prepped_sdp.json"
SAVE_PATH = "glove/smaller.json"

def get_dataset_counts():
    fo = open(DATASET, "r")
    dataset = json.load(fo)
    fo.close()

    all_tokens = []
    for d in dataset:
        all_tokens.extend(d["text"])
    counts = Counter(all_tokens)

    return counts

ALL_TOKENS = get_dataset_counts()

def load_glove():
    all_glove_embed = {}

    with open(GLOVE_PATH, "r") as f:
        parse_line = lambda x: (x[0], x[1:len(x)])
        lines = f.readlines()
        print("hihh")

        idx = 0
        for line in lines:
            word, vec = parse_line(line.split())
            all_glove_embed[word] = vec

            idx += 1
            if idx % 10000 == 0:
                print(idx)

    return all_glove_embed

def filter_embed(all_glove_embed):
    global ALL_TOKENS
    all_tokens_dataset = set(ALL_TOKENS)
    all_glove_words = set(all_glove_embed.keys())

    intersection_words = all_tokens_dataset.intersection(all_glove_words)
    print(len(intersection_words))
    smaller_glove = {}
    
    for word in intersection_words:
        smaller_glove[word] = all_glove_embed[word]

    assert len(smaller_glove.keys()) == len(intersection_words)
    
    return smaller_glove

print("hell")

fo = open(SAVE_PATH, "w+")
json.dump(filter_embed(load_glove()), fo)
fo.close()


