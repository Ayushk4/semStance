import torch
import json
from params import params
import random
import numpy as np
from torch_geometric.data import Data, Batch
from transformers import BertTokenizer

NORMALIZED_DATA_PATH = "data/data/normalized.json"

COMPANY_TO_NORMED = {"CVS": "cvs",
                    "AET": "avetna",
                    "CI": "cvgna",
                    "ESRX": "expresscripts",
                    "ANTM": "antema",
                    "HUM": "huumana"
                    }
NORMED_TO_COMPANY = {v:k for k,v in COMPANY_TO_NORMED.items()}

class wtwtDataset:
    def __init__(self):
        random.seed(params.python_seed)
        self.stance2id = {'comment': 0, 'unrelated': 1, 'support': 2, 'refute': 3}
        self.id2stance = {v: k for k,v in self.stance2id.items()}
        print(self.stance2id, "||", self.id2stance)
        self.num_edge_labels = 26
        test = []
        train_valid = []

        fo = open(NORMALIZED_DATA_PATH, "r")
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
            batch_text = []
            stances = []
            
            for single_tweet in unbatched[idx:min(idx+params.batch_size, num_data)]:
                this_stance_ids = self.stance2id[single_tweet["stance"]]
                criterion_weights[this_stance_ids] += 1
                stances.append(this_stance_ids)

                this_text_idx = []
                this_pad_masks = []
                this_segment_embed = []
                this_text = ""
                this_company = single_tweet["merger"].split('_')
                assert len(this_company) == 2
                flag = False
                for token in single_tweet["text"][1:]:                    
                    if token == "<target>":
                        wrd = " " + this_company[1]
                    elif token == "<buyer>":
                        wrd = " " + this_company[0]
                    elif token == "<sep>":
                        flag = True
                        break
                    else:
                        wrd = " " + token
                    this_text += " " + wrd
                assert flag == True

                this_text_2 = " " + this_company[0] + " " + this_company[1]
                batch_text.append([this_text, this_text_2])
                # print(this_text)

            tokenized_batch = self.bert_tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True,
                                                                return_tensors="pt", return_token_type_ids=True)

            texts = tokenized_batch['input_ids'].to(params.device)
            stances = torch.LongTensor(stances).to(params.device)
            pad_masks = tokenized_batch['attention_mask'].squeeze(1).to(params.device)
            segment_embed = tokenized_batch['token_type_ids'].to(params.device)

            # print("\n", texts[0, :], texts.size())
            # print("\n", stances, stances.size())
            # print("\n", pad_masks[0, :], pad_masks.size())
            # print("\n", segment_embed[0, :], segment_embed.size())

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)
            l = texts.size(1)
            assert texts.size() == torch.Size([b, l]) # Maxlen = 63 + 2 for CLS and SEP
            assert stances.size() == torch.Size([b])
            assert pad_masks.size() == torch.Size([b, l]) # Maxlen = 63 + 2 for CLS and SEP
            assert segment_embed.size() == torch.Size([b, l]) # Maxlen = 63 + 2 for CLS and SEP

            dataset.append((texts, stances, pad_masks, segment_embed))
            idx += params.batch_size

        print("num_batches=", len(dataset), " | num_data=", num_data)
        criterion_weights = np.sum(criterion_weights)/criterion_weights

        return dataset, criterion_weights/np.sum(criterion_weights)

if __name__ == "__main__":
    dataset = wtwtDataset()
    print("Train_dataset Size =", len(dataset.train_dataset),
            "Eval_dataset Size =", len(dataset.eval_dataset))
    print(dataset.train_dataset[0])
