# Post process text in graphs: function = normalize_text
    # Normalize
    #   - <user> back
    #   - <number>
    #   - <money>
    # Remove [?,.!|]
    # Lowercase
    # Converts buyer company name in text to <buyer>
    # Converts target's company's name in text to <target>
    # Handle apostrophe 's
    # Note Express Scripts has a new name
    # Also Uriah -> <user>
    # Removes DIS_FOXA
    # Pad

# Goal:
    # Concat of list over the tokens to tweet["text"];
    # Pad and create attention masks
    # Create extra vectors

# Pads text and creates attention masks
# Also Add <cls> at start and <sep> at end of sentence.

import glob
import json
import numpy as np

MERGER_TO_COMPANY = {
                    "CVS_AET": {"buyer": "cvs", "target": "avetna"},
                    "CI_ESRX": {"buyer": "cvgna", "target": "exprsx"},
                    "ANTM_CI": {"buyer": "antema", "target": "cvgna"},
                    "AET_HUM": {"buyer": "avetna", "target": "huumana"}
                    }

MAX_LEN = 0

NORMALIZE_MAPPING = {"Uriah": "<user>",
                        "1": "<money>",
                        "0": "<number>",
                        "'s": "s"
                    }

CCC = 0

def normalize_text(pad_len, tokens, merger, tweet_id, stance):
    # Normalize: <user>, <number>, <money>
    # Remove [?,.!|]
    # Lowercase
    # Handle apostrophe 's
    # Note Express Scripts has a new name

    # Converts buyer company name in text to <buyer>
    # Converts target's company's name in text to <target>

    # Pad
    normed_tokens = []
    for t in tokens:
        if t not in ["?", ",", ".", "!", "|"]:
            normed_tokens.append(NORMALIZE_MAPPING.get(t, t.lower()))
    # normed_tokens = [NORMALIZE_MAPPING.get(t, t.lower()) for t in tokens]
    buyer = merger["buyer"]
    target = merger["target"]

    vectors = [[0, 0]]
    new_toks = ["<cls>"]
    src_key_padding_mask = [False]
    found_buyer = False
    found_target = False

    idx = 0
    for token in normed_tokens:
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

    new_toks.append("<sep>")
    vectors.append([0, 0])
    src_key_padding_mask.append(False)

    assert (idx+2) <= pad_len
    i_ = idx

    while idx < pad_len:
        idx += 1
        new_toks.append("<paddd>")
        vectors.append([0, 0])
        src_key_padding_mask.append(True)

    if found_buyer == False and found_target == False:# and stance not in ["comment", "unrelated"]:
        global CCC
        CCC+=1
        # print(normed_tokens,
        #     buyer,
        # #     found_buyer,
        #     target,
        # #     found_target,
        #     tweet_id,
        #     stance,
        # #     # end=" "
        #     )

    assert len(normed_tokens) == i_
    assert len(vectors) == pad_len + 2 # +2 for <cls> and <sep>
    assert len(new_toks) == pad_len + 2
    assert len(src_key_padding_mask) == pad_len + 2

    return vectors, new_toks, [src_key_padding_mask] 


def srl_to_text(pad_len, graphs, merger, tweet_id, stance):
    tokens = []
    for sent in graphs:
        if type(sent) == list:
            tokens.extend(sent)
        else:
            tokens.extend(sent["words"])
    tokens, vectors, att_masks = normalize_text(pad_len, tokens, merger, tweet_id, stance)
    return tokens, vectors, att_masks


i = 0
dis_fox = 0
filtered_normealized = []
pad_len = 63
for fil_name in glob.glob("data/srl*"):
    fo = open(fil_name, "r")
    all_srl_data = json.load(fo)
    fo.close()

    for item in all_srl_data:
        # Removes DIS_FOXA
        if item["merger"] != "FOXA_DIS":
            new_item  = item.copy()
            vectors, new_tokens, src_key_pad_mask = srl_to_text(pad_len, 
                                                        item["srl_graphs"],
                                                        MERGER_TO_COMPANY[item["merger"]],
                                                        item["tweet_id"],
                                                        item["stance"])
            new_item["text"] = new_tokens
            new_item["extra_vectors"] = vectors
            new_item["pad_mask"] = src_key_pad_mask
            filtered_normealized.append(new_item)
            MAX_LEN = max(MAX_LEN, len(new_tokens))
            i += 1
        else:
            dis_fox += 1
    print(MAX_LEN, i, dis_fox, fil_name, "\n")

fo = open("data/normalized.json", "w+")
json.dump(filtered_normealized, fo, indent=2)
fo.close()

print(CCC, len(filtered_normealized))