import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--glove_dims", default=200, type=int)
parser.add_argument("--min_occur", default=10, type=int)
params = parser.parse_args()

DATASET_PATH = "sdp/prepped_sdp.json"
GLOVE_PREPARED_PATH = "glove/prepared.json"

IDX2WORD_SAVE_PATH = "glove/idx2word.json"
GLOVE_EMBED_SAVE_PATH = "glove/embed_glove.json"
DATASET_SAVE_PATH = "sdp/indexed.json"


# Load normalized dataset and smaller glove
fo = open(DATASET_PATH, "r")
dataset = json.load(fo)
fo.close()

fo = open(GLOVE_PREPARED_PATH, "r")
glove = json.load(fo)
fo.close()

for key, value in glove.items():
    glove[key] = [float(v) for v in value]

# Creat idx2word and glove_embed matrix
idx = 0
word2idx = {}
idx2word = {}
glove_embeds = []

for word in glove.keys():
    glove_embeds.append(glove[word])
    word2idx[word] = idx
    idx2word[idx] = word
    idx += 1

PAD = "<paddd>"
pad_idx = idx
glove_embeds.append([0.0] *len(glove_embeds[0]))
idx2word[pad_idx] = PAD
word2idx[PAD] = pad_idx

# Convert tokens to indexes
indexed_dataset = []
unk_idx = word2idx["<unknown>"]
for item in dataset:
    indexed_tokens = [word2idx.get(tok, unk_idx) for tok in item["text"]]

    indexed_item = item.copy()
    indexed_item["text"] = indexed_tokens
    indexed_item["raw_text"] = item["text"]
    indexed_dataset.append(indexed_item)

# Save idx2word, glove embed matrix, indexed dataset
print("Saving files")
def save_jason(obj, path):
    fo = open(path, "w+")
    json.dump(obj, fo)
    fo.close()

save_jason(idx2word, IDX2WORD_SAVE_PATH)
save_jason(glove_embeds, GLOVE_EMBED_SAVE_PATH)
save_jason(indexed_dataset, DATASET_SAVE_PATH)
