import torch
import json
from params import params
import random


class wtwtDataset:
    def __init__(self):
        random.seed(params.python_seed)
        self.stance2id = {'comment': 0, 'unrelated': 1, 'support': 2, 'refute': 3}
        self.id2stance = {v: k for k,v in self.stance2id.items()}
        print(self.stance2id, "||", self.id2stance)

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
            self.train_dataset = self.batched_dataset([train_valid[0]] * 2)
            self.eval_dataset = self.batched_dataset([train_valid[1]] * 2)
        else:
            if params.test_mode == False:
                split = (4 * len(train_valid)) // 5
                print("Train_dataset:", end= " ")
                self.train_dataset = self.batched_dataset(train_valid[:split])
                print("Valid_dataset:", end= " ")
                self.eval_dataset = self.batched_dataset(train_valid[split:])
            else:
                print("Full_Train_dataset:", end= " ")
                self.train_dataset = self.batched_dataset(train_valid)
                print("Test_dataset:", end= " ")
                self.eval_dataset = self.batched_dataset(test)

    def batched_dataset(self, unbatched): # For batching full or a part of dataset.
        dataset = []

        idx = 0
        num_data = len(unbatched)
        while idx < num_data:
            texts = []
            stances = []
            pad_masks = []
            target_buyr = []

            for single_tweet in unbatched[idx:min(idx+params.batch_size, num_data)]:
                texts.append(single_tweet["text"])
                stances.append(self.stance2id[single_tweet["stance"]])
                pad_masks.append(single_tweet["pad_mask"])
                target_buyr.append(single_tweet["extra_vectors"])

            texts = torch.LongTensor(texts)
            stances = torch.LongTensor(stances)
            pad_masks = torch.BoolTensor(pad_masks).permute(0, 2, 1)
            target_buyr = torch.Tensor(target_buyr).permute(0, 1, 2)

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)
            # print(b)
            assert texts.size() == torch.Size([b, 72]) # Maxlen = 70 + 2 for CLS and SEP
            assert stances.size() == torch.Size([b])
            assert pad_masks.size() == torch.Size([b, 72, 1]) # Maxlen = 70 + 2 for CLS and SEP
            assert target_buyr.size() == torch.Size([b, 72, 2]) # Maxlen = 70 + 2 for CLS and SEP

            # print("\n", texts, texts.size())
            # print("\n", stances, stances.size())
            # print("\n", pad_masks, pad_masks.size())
            # print("\n", target_buyr, target_buyr.size())
            
            dataset.append((texts, stances, pad_masks, target_buyr))
            idx += params.batch_size

        print("num_batches=", len(dataset), " | num_data=", num_data)

        return dataset
        
if __name__ == "__main__":
    dataset = wtwtDataset()
    print("Train_dataset Size =", len(dataset.train_dataset),
            "Eval_dataset Size =", len(dataset.eval_dataset))
