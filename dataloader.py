import torch
import json
from params import params
import random
import numpy as np

TEXT_PADDED_LEN = 65
ROOT_NODES_PADDED_LEN = 15
CHILD_NODES_PADDED_LEN = 36
EDGE_LABEL_PADDED_LEN = CHILD_NODES_PADDED_LEN

class wtwtDataset:
    def __init__(self):
        random.seed(params.python_seed)
        self.stance2id = {'comment': 0, 'unrelated': 1, 'support': 2, 'refute': 3}
        self.id2stance = {v: k for k,v in self.stance2id.items()}
        print(self.stance2id, "||", self.id2stance)
        self.num_edge_labels = 26
        test = []
        train_valid = []

        fo = open(params.dataset_path, "r")
        full_dataset = json.load(fo)
        fo.close()

        for data in full_dataset:
            if data["merger"] == params.target_merger:
                test.append(data)
            else:
                train_valid.append(data)

        print("Before shuffling[0] = ", train_valid[0]["tweet_id"], end=" | ")
        random.shuffle(train_valid)
        print("After shuffling[0] = ", train_valid[0]["tweet_id"])

        if params.dummy_run == True:
            self.train_dataset, self.criterion_weights = self.batched_dataset([train_valid[0]] * 2)
            self.eval_dataset, _ = self.batched_dataset([train_valid[1]] * 2)
        else:
            if params.test_mode == False:
                split = (4 * len(train_valid)) // 5
                print("Train_dataset:", end= " ")
                self.train_dataset, self.criterion_weights = self.batched_dataset(train_valid[:split])
                print("Valid_dataset:", end= " ")
                self.eval_dataset, _ = self.batched_dataset(train_valid[split:])
            else:
                print("Full_Train_dataset:", end= " ")
                self.train_dataset, self.criterion_weights = self.batched_dataset(train_valid)
                print("Test_dataset:", end= " ")
                self.eval_dataset, _ = self.batched_dataset(test)

        self.criterion_weights = torch.tensor(self.criterion_weights.tolist()).to(params.device)
        print("Training loss weighing = ", self.criterion_weights)

    def batched_dataset(self, unbatched): # For batching full or a part of dataset.
        dataset = []
        criterion_weights = np.zeros(4) + 0.0000001 # 4 labels 

        idx = 0
        num_data = len(unbatched)

        while idx < num_data:
            texts = []
            stances = []
            pad_masks = []
            target_buyr = []

            edge_labels = []
            lstm_root_masks = []
            lstm_child_masks = []
            gatt_masks_for_root = []
            gatt_root_idxs = []
            semantics_root_mask = []

            for single_tweet in unbatched[idx:min(idx+params.batch_size, num_data)]:
                texts.append(single_tweet["text"])
                this_stance_ids = self.stance2id[single_tweet["stance"]]
                criterion_weights[this_stance_ids] += 1
                stances.append(this_stance_ids)
                pad_masks.append(single_tweet["pad_mask"])
                target_buyr.append(single_tweet["extra_vectors"])
                
                edge_labels.append(single_tweet["edge_labels"])
                lstm_root_masks.append(single_tweet["root_masks_over_lstm"])
                lstm_child_masks.append(single_tweet["child_masks_over_lstm"])
                gatt_masks_for_root.append(single_tweet["child_masks_for_root_att"])
                gatt_root_idxs.append(single_tweet["root_idxs_for_child_att"])
                semantics_root_mask.append(single_tweet["sem_graph_pool_mask"])

            texts = torch.LongTensor(texts).to(params.device)
            stances = torch.LongTensor(stances).to(params.device)
            pad_masks = torch.BoolTensor(pad_masks).squeeze(1).to(params.device)
            target_buyr = torch.Tensor(target_buyr).to(params.device)

            edge_labels = torch.LongTensor(edge_labels).to(params.device)
            lstm_root_masks =  ~torch.BoolTensor(lstm_root_masks).to(params.device)
            lstm_child_masks = ~torch.BoolTensor(lstm_child_masks).to(params.device)
            gatt_masks_for_root = ~torch.BoolTensor(gatt_masks_for_root).to(params.device)
            gatt_root_idxs = torch.LongTensor(gatt_root_idxs).to(params.device)
            semantics_root_mask = ~torch.BoolTensor(semantics_root_mask).to(params.device)

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)

            assert texts.size() == torch.Size([b, TEXT_PADDED_LEN]) # Maxlen = 63 + 2 for CLS and SEP
            assert stances.size() == torch.Size([b])
            assert pad_masks.size() == torch.Size([b, TEXT_PADDED_LEN]) # Maxlen = 63 + 2 for CLS and SEP
            assert target_buyr.size() == torch.Size([b, TEXT_PADDED_LEN, 2]) # Maxlen = 63 + 2 for CLS and SEP

            assert edge_labels.size() == torch.Size([b, EDGE_LABEL_PADDED_LEN])
            assert lstm_root_masks.size() == torch.Size([b, ROOT_NODES_PADDED_LEN, TEXT_PADDED_LEN])
            assert lstm_child_masks.size() == torch.Size([b, CHILD_NODES_PADDED_LEN, TEXT_PADDED_LEN])
            assert gatt_masks_for_root.size() == torch.Size([b, ROOT_NODES_PADDED_LEN, CHILD_NODES_PADDED_LEN])
            assert gatt_root_idxs.size() == torch.Size([b, CHILD_NODES_PADDED_LEN])
            assert semantics_root_mask.size() == torch.Size([b, ROOT_NODES_PADDED_LEN])

            # print("\n", texts, texts.size())
            # print("\n", stances, stances.size())
            # print("\n", pad_masks, pad_masks.size())
            # print("\n", target_buyr, target_buyr.size())

            dataset.append((texts, stances, pad_masks, target_buyr, edge_labels,
                        lstm_root_masks, lstm_child_masks, gatt_masks_for_root,
                        gatt_root_idxs, semantics_root_mask
            ))
            idx += params.batch_size
        # HANDLE CASES WITH NO NODES IN SEM GRAPH => Already handled.
        print("num_batches=", len(dataset), " | num_data=", num_data)
        assert edge_labels[0][-1] == self.num_edge_labels # And accordingly change in the model specification
        criterion_weights = np.sum(criterion_weights)/criterion_weights
 
        return dataset, criterion_weights/np.sum(criterion_weights)

if __name__ == "__main__":
    dataset = wtwtDataset()
    print("Train_dataset Size =", len(dataset.train_dataset),
            "Eval_dataset Size =", len(dataset.eval_dataset))
