import torch
import json
from params import params
import random
import numpy as np
from torch_geometric.data import Data, Batch
from transformers import BertTokenizer
import pickle as pkl
import os

COMPANY_TO_NORMED = {"CVS": "cvs",
                    "AET": "aetna",
                    "CI": "cigna",
                    "ESRX": "expresscripts",
                    "ANTM": "antema",
                    "HUM": "huumana"
                    }

NORMED_TO_COMPANY = {v:k for k,v in COMPANY_TO_NORMED.items()}
TEXT_PADDED_LEN = 92

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

        self.bert_tokenizer = BertTokenizer.from_pretrained(params.bert_type)
        new_special_tokens_dict = {"additional_special_tokens": ["<number>", "<money>", "<user>"]}
        self.bert_tokenizer.add_special_tokens(new_special_tokens_dict)
        print("Loaded Bert Tokenizer")

        if params.dummy_run == True:
            self.train_dataset, self.criterion_weights = self.batched_dataset([train_valid[0]] * 2)
            self.eval_dataset, _ = self.batched_dataset([train_valid[1]] * 2)
        else:
            if params.test_mode != "True":
                split = (len(train_valid)) // 5
                valid_num = params.cross_valid_num
                if valid_num == 4:
                    split *= 4
                    print("Train_dataset:", end= " ")
                    self.train_dataset, self.criterion_weights = self.batched_dataset(train_valid[:split])
                    print("Valid_dataset:", end= " ")
                    self.eval_dataset, _ = self.batched_dataset(train_valid[split:])
                elif valid_num == 0:
                    print("Train_dataset:", end= " ")
                    self.train_dataset, self.criterion_weights = self.batched_dataset(train_valid[split:])
                    print("Valid_dataset:", end= " ")
                    self.eval_dataset, _ = self.batched_dataset(train_valid[:split])
                else:
                    print("Train_dataset:", end= " ")
                    self.train_dataset, self.criterion_weights = self.batched_dataset(train_valid[:(split * valid_num)] + train_valid[(split * (valid_num+1)):])
                    print("Valid_dataset:", end= " ")
                    self.eval_dataset, _ = self.batched_dataset(train_valid[(split * valid_num):(split * (valid_num+1))])
            else:
                potential_pkl_path = params.bert_type + "_" + params.target_merger + ".pkl"
                if potential_pkl_path in os.listdir("."):
                    self.train_dataset, self.criterion_weights, self.eval_dataset = \
                            pkl.load(open(potential_pkl_path, "rb"))
                else: # not present
                    print("Full_Train_dataset:", end= " ")
                    self.train_dataset, self.criterion_weights = self.batched_dataset(train_valid)
                    print("Test_dataset:", end= " ")
                    self.eval_dataset, _ = self.batched_dataset(test)
                    pkl.dump((self.train_dataset, self.criterion_weights, self.eval_dataset),
                                open(potential_pkl_path, "wb+"))

        self.criterion_weights = torch.tensor(self.criterion_weights.tolist()).to(params.device)
        print("Training loss weighing = ", self.criterion_weights)

    def batched_dataset(self, unbatched): # For batching full or a part of dataset.
        dataset = []
        criterion_weights = np.zeros(4) + 0.0000001 # 4 labels 

        num_data = len(unbatched)
        companies_normed = NORMED_TO_COMPANY.keys()
        pad_idx = self.bert_tokenizer.convert_tokens_to_ids(
                        self.bert_tokenizer.tokenize("[PAD]"))[0]

        idx = 0
        while idx < num_data:
            texts = []
            stances = []

            edge_indices = []
            edge_labels = []
            num_edges_in_batch = 0
            for single_tweet in unbatched[idx:min(idx+params.batch_size, num_data)]:
                num_edges_in_batch += len(single_tweet["edges"])
                this_company = single_tweet["merger"].split('_')

                this_stance_ids = self.stance2id[single_tweet["stance"]]
                criterion_weights[this_stance_ids] += 1
                stances.append(this_stance_ids)

                this_word_idx_to_subword_idx = {}
                this_text = []
                for t_idx, token in enumerate(single_tweet["raw_text"]):
                    if token == "<cls":
                        this_token = "[CLS]"
                    elif token == "<sep>":
                        this_token = "[SEP]"
                        this_word_idx_to_subword_idx[t_idx] = len(this_text)
                        this_text.extend(self.bert_tokenizer.convert_tokens_to_ids(
                                        self.bert_tokenizer.tokenize(this_token)))
                        # Add context
                        this_token = this_company[0] + " buys " + this_company[1] + " [SEP]"
                        this_text.extend(self.bert_tokenizer.convert_tokens_to_ids(
                                        self.bert_tokenizer.tokenize(this_token)))
                        # pad to maxlen
                        lens = TEXT_PADDED_LEN - len(this_text)
                        assert lens >= 0
                        this_text += ([pad_idx] * lens)
                        break
                    elif token == "<paddd>":
                        raise
                    elif token == "<target>":
                        this_token = this_company[1]
                    elif token == "<buyer>":
                        this_token = this_company[0]
                    elif token in companies_normed:
                        this_token = NORMED_TO_COMPANY[token]
                    else:
                        this_token = token
                    this_word_idx_to_subword_idx[t_idx] = len(this_text)
                    this_text.extend(self.bert_tokenizer.convert_tokens_to_ids(
                                        self.bert_tokenizer.tokenize(this_token)))

                texts.append(this_text)
                w_idx2_sw_ix = this_word_idx_to_subword_idx
                if len(single_tweet["edges"]) == 0:
                    edge_indices.append(torch.LongTensor([[], []]))
                    edge_labels.append(torch.LongTensor([]))
                else:
                    this_edge_indices = torch.LongTensor([[w_idx2_sw_ix[x[0]], w_idx2_sw_ix[x[1]]] for x in single_tweet["edges"]]).t().contiguous()
                    assert this_edge_indices.size() == torch.Size([2, len(single_tweet["edges"])])
                    edge_indices.append(this_edge_indices)

                    this_edge_label = torch.LongTensor([x[3] for x in single_tweet["edges"]])
                    assert this_edge_label.size() == torch.Size([len(single_tweet["edges"])])
                    edge_labels.append(this_edge_label)
            if idx < 1: # Just print one.
                print(self.bert_tokenizer.convert_ids_to_tokens(this_text),
                    single_tweet["raw_text"], edge_indices, w_idx2_sw_ix, single_tweet["edges"], "\n\n")
            texts = torch.LongTensor(texts).to(params.device)
            stances = torch.LongTensor(stances).to(params.device)

            dummy_x = torch.randn(TEXT_PADDED_LEN, 1)
            edge_indices = Batch.from_data_list([Data(x=dummy_x, edge_index=edge_index.to(params.device)) for edge_index in edge_indices]).edge_index
            edge_labels = [edge_label.to(params.device) for edge_label in edge_labels]
            edge_labels = torch.cat(edge_labels, 0)

            from collections import Counter
       	    counts = dict(Counter(edge_indices.tolist()[1]))
            edge_weights = torch.tensor([1/counts[x] for x in edge_indices.tolist()[1]])
            # edge_masks wrt 1, [1] index is the i node, softmax for all its neighbours
            edge_masks = [(edge_indices[1, :] == i) for i in edge_indices[1, :].tolist()]
            edge_masks = torch.stack(edge_masks)
            # Additive edge masks needed for torch v1.2
            #edge_masks = torch.zeros(edge_masks.shape).to(edge_masks.device).masked_fill(~edge_masks, -10000.0)

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)
            assert texts.size() == torch.Size([b, TEXT_PADDED_LEN]) # Maxlen = 63 + 2 for CLS and SEP
            assert stances.size() == torch.Size([b])

            e_num = num_edges_in_batch
            assert edge_indices.size() == torch.Size([2, e_num])
            assert edge_labels.size() == torch.Size([e_num])
            assert edge_weights.size() == torch.Size([e_num])
            assert edge_masks.size() == torch.Size([e_num, e_num])

            # print("\n", texts, texts.size())
            # print("\n", stances, stances.size())
            # print("\n", pad_masks, pad_masks.size())
            # print("\n", target_buyr, target_buyr.size())
            # print("\n", edge_masks[:20, :20], edge_indices)

            dataset.append((texts, stances, None, None, edge_indices, edge_labels, edge_weights, edge_masks))
            idx += params.batch_size

        # HANDLE CASES WITH NO NODES IN SEM GRAPH => Already handled.
        print("num_batches=", len(dataset), " | num_data=", num_data)
        criterion_weights = np.sum(criterion_weights)/criterion_weights

        return dataset, criterion_weights/np.sum(criterion_weights)

if __name__ == "__main__":
    dataset = wtwtDataset()
    print("Train_dataset Size =", len(dataset.train_dataset),
            "Eval_dataset Size =", len(dataset.eval_dataset))
    print(dataset.train_dataset[0])
    #while True:
    #    pass
