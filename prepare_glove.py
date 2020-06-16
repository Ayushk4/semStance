# Ip: glove/smaller.json
# Op: glove/prepared.json

# Adds to vocab, the tokens with more than 10 occurrences
# Handle special tokens.

import argparse
import json
from collections import Counter
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--glove_dims", default=200, type=int)
parser.add_argument("--min_occur", default=10, type=int)
params = parser.parse_args()

DATASET = "srl/data/normalized.json"
GLOVE_SMALLER_PATH = "glove/smaller.json"
SAVE_PATH = "glove/prepared.json"

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

fo = open(GLOVE_SMALLER_PATH, "r")
glove = json.load(fo)
fo.close()

print(len(ALL_TOKENS), len(glove.keys()))

special_tokens = ["<buyer>", "<target>", "<number>", "<user>", "<money>", "<cls>", "<sep>", "<paddd>"]
COMPANY_NAMES = ["cvs", "avetna", "cvgna", "expresscripts", "antema",  "huumana"]

for tok in ALL_TOKENS:
    if tok not in COMPANY_NAMES and \
        ALL_TOKENS[tok] > params.min_occur and \
        (tok in special_tokens or tok not in glove.keys()): # We reinitialize special tokens
        if tok == "<paddd>": # We don't need no embedding for pad
            print(glove.pop(tok, "  padd not present"))
        else:
            glove[tok] = list(np.random.uniform(low=-0.1, high=0.1, size=params.glove_dims))
            # print(tok, "\t", ALL_TOKENS[tok], "\t", "%.4f" % glove[tok][0])

print(len(ALL_TOKENS), len(glove.keys()))

glove["<unknown>"] = list(np.random.uniform(low=-0.1, high=0.1, size=params.glove_dims))

fo = open(SAVE_PATH, "w+")
json.dump(glove, fo)
fo.close()
