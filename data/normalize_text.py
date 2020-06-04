import json
import torch
import numpy as np
# Does not normalize, basically converts the following 2 things
  # Converts buyer company name in text to <buyer>
  # Converts target's company's name in text to <target>
# Removes DIS_FOXA
# Add concat list of list over the tokens.
# Pads text and creates attention masks
# Also Add <cls> at start and <sep> at end of sentence.

MERGER_TO_COMPANY = {
                    "CVS_AET": {"buyer": "cvs", "target": "avetna"},
                    "CI_ESRX": {"buyer": "cvgna", "target": "expresscripts"},
                    "ANTM_CI": {"buyer": "antema", "target": "cvgna"},
                    "AET_HUM": {"buyer": "avetna", "target": "huumana"}
                    }
CCC = 0

def embeds_and_normalize(pad_len, tokens, merger, tweet_id, stance): # stance and tweet_id only for debugging
    buyer = merger["buyer"]
    target = merger["target"]

    vectors = [[0, 0]]
    new_toks = ["<cls"]
    src_key_padding_mask = [False]
    found_buyer = False
    found_target = False

    idx = 0
    for token in tokens:
        if token == buyer:
            vectors.append([1, 0])
            new_toks.append("<buyer>")
            found_buyer = True
        elif token == target:
            vectors.append([0, 1])
            new_toks.append("<target>")
            found_target = True
        else:
            new_toks.append(token)
            vectors.append([0, 0])        

        src_key_padding_mask.append(False)
        idx+=1

    assert idx <= pad_len
    i_ = idx

    new_toks.append("<sep>")
    vectors.append([0, 0])
    src_key_padding_mask.append(False)

    while idx < pad_len:
        idx += 1
        new_toks.append("<paddd>")
        vectors.append([0, 0])
        src_key_padding_mask.append(True)

    if found_buyer == False and found_target == False and stance == "unrelated":
        global CCC
        CCC+=1
    #     print(tokens,
    #         buyer,
    #         found_buyer,
    #         target,
    #         found_target,
    #         tweet_id,
    #         stance,
    #         # end=" "
    #         )

    assert len(tokens) == i_
    assert len(vectors) == pad_len + 2 # +2 for <cls> and <sep>
    assert len(new_toks) == pad_len + 2
    assert len(src_key_padding_mask) == pad_len + 2

    return vectors, new_toks, [src_key_padding_mask]

print(embeds_and_normalize(70, "cvs to buy avetna for <money> billion in a deal that may reshape the health industry".split(" "),
                                MERGER_TO_COMPANY["CVS_AET"],
                                "", ""))

fo = open("data/pre_processed.json", "r")
pre_proced = json.load(fo)
fo.close()

# Finding max sequence length for padding
max_len = 0
for item in pre_proced:
    if item["merger"] != "FOXA_DIS":
        if len(item["text"]) > max_len:
            max_len = len(item["text"])
            print(item["tweet_id"], max_len)

filtered_normealized = []
pad_len = max_len + 1
for item in pre_proced:
    if item["merger"] != "FOXA_DIS":
        new_item  = item.copy()
        vectors, new_tokens, src_key_pad_mask = embeds_and_normalize(pad_len,
                                            item["text"],
                                            MERGER_TO_COMPANY[item["merger"]],
                                            item["tweet_id"],
                                            item["stance"]
                                            )
        new_item["text"] = new_tokens
        new_item["extra_vectors"] = vectors
        new_item["pad_mask"] = src_key_pad_mask
        filtered_normealized.append(new_item)

fo = open("data/normalized.json", "w+")
json.dump(filtered_normealized, fo, indent=2)
fo.close()

print(CCC, len(filtered_normealized))
