# Makes the graphs batching ready by appropriate padding

max_roots = 15
max_childs = 36
pad_len = 65
PAD_LABEL = "<PAD_LABEL>"

ALL_LABELS = []
EMPTY = 0
def make_batchable(labels, lstm_root_masks, lstm_child_masks, gatt_masks_for_root, gatt_root_idxs):
    assert len(labels) < max_childs
    assert len(lstm_child_masks) < max_childs
    assert len(lstm_root_masks) < max_roots
    assert len(gatt_masks_for_root) < max_roots
    assert len(gatt_root_idxs) < max_childs
    ALL_LABELS.extend(labels)

    semantics_aug_mask = ([1] * len(lstm_root_masks)) + ([0] * (max_roots - len(lstm_root_masks)))
    assert sum(semantics_aug_mask) == len(lstm_root_masks)
    if (len(lstm_root_masks)) == 0:
        EMPTY+=1
    assert len(semantics_aug_mask) == max_roots

    while True:
        if len(labels) == max_childs:
            break
        labels.append(PAD_LABEL)
    # print("edge labels padded")

    while True:
        if len(lstm_child_masks) == max_childs:
            break
        lstm_child_masks.append(([0] * (pad_len-1)) + [1])
    # print("lstm child masks padded")

    while True:
        if len(lstm_root_masks) == max_roots:
            break
        lstm_root_masks.append(([0] * (pad_len-1)) + [1])
    # print("lstm root masks padded")

    while True:
        if len(gatt_root_idxs) == max_childs:
            break
        gatt_root_idxs.append(max_roots-1)
    # print("gatt root idx padded")

    new_gatt_masks_for_root = []
    for this_root_idx_mask in gatt_masks_for_root:
        l = len(this_root_idx_mask)
        this_root_idx_mask.extend([0] * (max_childs - l))
        new_gatt_masks_for_root.append(this_root_idx_mask)

    while True:
        if len(new_gatt_masks_for_root) == max_roots:
            break
        new_gatt_masks_for_root.append(([0] * (max_childs-1)) + [1])
    # print("gatt root idx padded")

    gatt_masks_for_root = new_gatt_masks_for_root
    assert len(list(set([len(ng) for ng in gatt_masks_for_root]))) == 1
    assert len(gatt_masks_for_root[0]) == max_childs

    assert len(labels) == max_childs
    assert len(lstm_child_masks) == max_childs
    assert len(lstm_root_masks) == max_roots
    assert len(gatt_masks_for_root) == max_roots
    assert len(gatt_root_idxs) == max_childs

    return labels, lstm_root_masks, lstm_child_masks, gatt_masks_for_root, gatt_root_idxs, semantics_aug_mask 

import json
fo = open("data/mapped.json", "r")
normalized_data = json.load(fo)
fo.close()

batchable = []
i=0
print("Starting")
for item in normalized_data:
    new_item = item.copy()
    batchable_graph = make_batchable(item["edge_labels"],
                                    item["root_masks_over_lstm"],
                                    item["child_masks_over_lstm"],
                                    item["child_masks_for_root_att"],
                                    item["root_idxs_for_child_att"])
    labels, lstm_root_masks, lstm_child_masks, gatt_masks_for_root, gatt_root_idxs, sem_aug_mask = batchable_graph

    new_item["edge_labels"] = labels
    new_item["root_masks_over_lstm"] = lstm_root_masks
    new_item["child_masks_over_lstm"] = lstm_child_masks
    new_item["child_masks_for_root_att"] = gatt_masks_for_root
    new_item["root_idxs_for_child_att"] = gatt_root_idxs
    new_item["sem_graph_pool_mask"] = sem_aug_mask

    batchable.append(new_item)

    if i%1000 == 0:
        print(i)
    i+=1

print("Total=", i)
ALL_LABELS = list(set(ALL_LABELS))
ALL_LABELS.append(PAD_LABEL)

i = 0
label_mapping = {}
for labl in ALL_LABELS:
    label_mapping[labl] = i
    i += 1

for i in range(len(batchable)):
    batchable[i]["edge_labels"] = list(map(lambda x: label_mapping[x], batchable[i]["edge_labels"]))

print("Edge Label -> number Done")
print("EMPTY = ", EMPTY)
fo = open("data/batchable.json", "w+")
json.dump(batchable, fo)
fo.close()
