# Remove [?,.!|] while mapping
# Pad
# Add 1 for CLS and SEP
# Handle apostrophe 's

# Goal:
    # - root_masks;
    # - child_masks;
    # - edge_labels (torch.long);
    # - child_to_root_labels (for h_child)
    # - root to child masks

import json
pad_len = 65 # 63 + <cls> + <sep>
COMPANY_NAMES = ["Cvs", "Avetna", "Cvgna", "Exprsx", "Antema", "Huumana",
                "cvs", "avetna", "cvgna", "exprsx", "antema", "huumana"]
IGNORE_VERBS = ['may', 'can', 'will', 'could', 'would', 'might', 'should']

def normalize_tags(tags):
    assert type(tags) == list
    normalized_tags = []

    for tag in tags:
        if tag == "O":
            normalized_tags.append(tag)
            continue
        tag = tag[2:]
        if "ARG" in tag and tag[-1].isdigit():
            tag = tag[:-1] +"0" 
        normalized_tags.append(tag)

    return normalized_tags

# edge_labels_cnt = ['I-R-ARGM-LOC', 'I-ARGM-PRP', 'B-ARGM-MOD', 'B-ARGM-PRP', 'I-ARGM-COM', 'B-R-ARGM-TMP',
#                 'I-R-ARGM-MNR', 'B-R-ARGM-CAU', 'B-ARGM-PRD', 'I-ARGM-ADJ', 'B-ARG3', 'B-ARGM-LOC', 'B-ARG5',
#                 'I-ARG4', 'I-R-ARG1', 'I-V', 'I-ARGM-MOD', 'B-ARGM-EXT', 'B-ARGM-ADJ', 'B-ARG4', 'I-ARGM-NEG',
#                 'B-C-ARG1', 'B-ARG2', 'B-ARGM-CAU', 'I-ARGM-DIS', 'I-R-ARG3', 'B-ARGM-TMP', 'B-ARGM-GOL', 'B-ARGM-NEG',
#                 'B-R-ARG2', 'B-ARGM-COM', 'B-R-ARG3', 'B-ARGM-MNR', 'I-ARGM-LOC', 'O', 'B-ARGM-DIS', 'I-R-ARG0', 'B-C-ARG2',
#                 'I-ARGM-PNC', 'I-ARG3', 'B-ARGM-REC', 'I-C-ARGM-ADV', 'I-ARGM-EXT', 'B-C-ARGM-ADV', 'B-ARG0', 'B-C-ARG0',
#                 'B-ARG1', 'I-ARG2', 'B-ARGM-LVB', 'I-ARGM-ADV', 'B-ARGM-ADV', 'I-ARGM-CAU', 'B-ARGM-DIR', 'I-ARGM-DIR',
#                 'B-V', 'B-R-ARG1', 'I-ARGM-PRD', 'B-R-ARGM-MNR', 'I-C-ARG0', 'I-ARG0', 'B-R-ARG0', 'B-R-ARGM-LOC',
#                 'I-ARGM-MNR', 'I-ARGM-GOL', 'I-ARGM-TMP', 'B-ARGM-PNC', 'I-ARG1', 'I-C-ARG2', 'I-C-ARG1']
# edge_labels_cnt = {label: 0 for label in list(set(normalize_tags(edge_labels_cnt)))}
ll = 0

def parse_label_seq(tweet_idx, label_sequence):
    edges = []
    child_idx_range = []
    root_idx_range = [-1, -1]
    idx = 0

    while idx < len(label_sequence):
        if label_sequence[idx] == "O":
            idx += 1
        elif label_sequence[idx] == "V":
            root_idx_range[0] = idx
            idx += 1
            while idx < len(label_sequence) and label_sequence[idx] == "V":
                idx += 1
            root_idx_range[1] = idx
        else:
            label = label_sequence[idx]
            child_range = [-1, -1]
            child_range[0] = idx
            idx += 1
            while idx < len(label_sequence) and label_sequence[idx] == label:
                idx += 1
            child_range[1] = idx
            child_idx_range.append(child_range)
            edges.append(label)

    if root_idx_range[0] == -1:
        print(tweet_idx, label_sequence)
    assert root_idx_range[0] != -1
    assert root_idx_range[1] != -1
    assert root_idx_range[0] < root_idx_range[1]
    assert len(edges) == len(child_idx_range)
    assert len(edges) > 0

    return root_idx_range, edges, child_idx_range

def graph_mapping_create(graphs, tokens, t_id):
    # Remove [?,.!|] while mapping
    # Pad
    # Add 1 for CLS and SEP
    assert pad_len == len(tokens)
    # print(tokens)
    # print(graphs)
    edge_labels = []
    root_masks = []
    child_masks = []
    root_to_child_masks = [] # Also equals root_to_edge_vectors
    child_to_root_idxs = []

    idx = 1 # Starts with <cls>
    for sentence in graphs:
        if type(sentence) == list:
            words = sentence
            for w in words:
                if w not in ["?", ",", ".", "!", "|"]:
                    idx += 1
        else:
            idx_cache = idx
            words = sentence["words"]

            labels = []
            for tree in sentence["verbs"]:
                if tree["verb"].lower() in COMPANY_NAMES or \
                        tree["verb"].lower() in IGNORE_VERBS or \
                        len(list(set(tree["tags"]))) <= 2:
                    continue
                elif "B-V" not in tree["tags"]:
                    print(tree)
                    continue
                else:
                    labels.append(normalize_tags(tree["tags"]))

            processed_label_seq = [[]]*len(labels)
            for w_idx in range(len(words)):
                w = words[w_idx]
                if w not in ["?", ",", ".", "!", "|"]:
                    for i in range(len(labels)):
                        t = processed_label_seq[i].copy()
                        t.append(labels[i][w_idx])
                        processed_label_seq[i] = t
                    idx += 1

            if len(labels) == 0:
                continue

            assert len(list(set([len(p) for p in processed_label_seq]))) == 1 # i.e. all tags of same length
            assert idx == idx_cache + len(processed_label_seq[0])

            for p_label_seq in processed_label_seq:
                root_idx_range, edges, child_idx_ranges = parse_label_seq(idx_cache, p_label_seq)

                this_root_mask = ([0] * (idx_cache + root_idx_range[0])) + \
                                ([1] * (root_idx_range[1] - root_idx_range[0])) + \
                                ([0] * (pad_len - (root_idx_range[1] + idx_cache)))
                assert sum(this_root_mask) > 0
                root_masks.append(this_root_mask)

                edge_labels.extend(edges)
                start_child_idx = len(child_masks)
                for child_idx_range in child_idx_ranges:
                    this_child_mask = ([0] * (idx_cache + child_idx_range[0])) + \
                                ([1] * (child_idx_range[1] - child_idx_range[0])) + \
                                ([0] * (pad_len - (child_idx_range[1] + idx_cache)))
                    assert sum(this_child_mask) > 0
                    child_masks.append(this_child_mask)
                end_child_idx = len(child_masks)

                assert start_child_idx < end_child_idx

                root_to_child_mask = ([0] * (start_child_idx)) + \
                                ([1] * (end_child_idx - start_child_idx))
                root_to_child_masks.append(root_to_child_mask)

                child_to_root_idxs.extend([len(root_masks)-1] * (end_child_idx - start_child_idx))

    # equalize_length(root_to_child_masks)
    # print(child_to_root_idxs, root_to_child_masks)
    assert tokens[idx] == "<sep>"
    assert len(edge_labels) == len(child_masks)
    assert len(edge_labels) >= len(root_masks)
    assert len(child_to_root_idxs) == len(edge_labels)
    assert len(root_to_child_masks) == len(root_masks)
    # print(idx)
    return edge_labels, root_masks, child_masks, root_to_child_masks, child_to_root_idxs

fo = open("data/normalized.json", "r")
normalized_data = json.load(fo)
fo.close()

print("start")
final_data = []
i = 0
for item in normalized_data:
    new_item = item.copy()
    graph_mappings = graph_mapping_create(new_item["srl_graphs"],
                                    new_item["text"],
                                    new_item["tweet_id"])
    edge_labels, root_masks, child_masks, root_to_child_masks, child_to_root_idxs = graph_mappings
    # print(edge_labels, root_masks, child_masks, root_to_child_masks, child_to_root_idxs, sep="\n")
    new_item["edge_labels"] = edge_labels
    new_item["root_masks_over_lstm"] = root_masks
    new_item["child_masks_over_lstm"] = child_masks
    new_item["child_masks_for_root_att"] = root_to_child_masks
    new_item["root_idxs_for_child_att"] = child_to_root_idxs

    final_data.append(new_item)
    i += 1
    if i%1000 == 0:
        print(i)

fo = open("data/mapped.json", "w+")
json.dump(final_data, fo, indent=2)
fo.close()

