# Integrate wandb
# Create data_loader instance
# Create_model instance on gpu
# Define Optimizer and loss
# Train and eval functions
# DataParallel
# Watch model on wandb, upload config on wandb

from params import params
import wandb
if params.wandb:
    wandb.init(project="semstance", name=params.run)
    wandb.config.update(params)

import torch
from bertloader import wtwtDataset

from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import BertModel, BertPreTrainedModel, BertConfig
import torch, torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import json
torch.manual_seed(params.torch_seed)

def train(model, dataset, criterion):
    model.train()
    train_losses = []
    num_batch = 0

    for batch in dataset:
        (texts, stances, att_masks, token_type) = batch
        preds = model(texts, att_masks, token_type)
        loss = criterion(preds, stances)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
      #  scheduler.step()

        if num_batch % 100 == 0:
            print("Train loss at {}:".format(num_batch), loss.item())

        num_batch += 1
        train_losses.append(loss.item())

    return np.average(train_losses)

def evaluate(model, dataset, criterion, target_names):
    model.eval()
    valid_losses = []
    predicts = []
    gnd_truths = []

    with torch.no_grad():
        for batch in dataset:
            (texts, stances, att_masks, token_type) = batch
            preds = model(texts, att_masks, token_type)

            loss = criterion(preds, stances)

            predicts.extend(torch.max(preds, axis=1)[1].tolist())
            gnd_truths.extend(stances.tolist())
            valid_losses.append(loss.item())

    assert len(predicts) == len(gnd_truths)

    confuse_mat = confusion_matrix(gnd_truths, predicts)
    if params.dummy_run:
        classify_report = {"hi": {"fake": 1.2}}
    else:
        classify_report = classification_report(gnd_truths, predicts, target_names=target_names, output_dict=True)

    mean_valid_loss = np.average(valid_losses)
    print("Valid_loss", mean_valid_loss)
    print(confuse_mat)

    for labl in target_names:
        print(labl,"F1-score:", classify_report[labl]["f1-score"])
    print("Accu:", classify_report["accuracy"])
    print("F1-Weighted", classify_report["weighted avg"]["f1-score"])
    print("F1-Avg", classify_report["macro avg"]["f1-score"])

    return mean_valid_loss, confuse_mat ,classify_report

########## Loading Glove ############
def open_it(pth):
    fo = open(pth, "r")
    j = json.load(fo)
    fo.close()
    return j
glove_embed = open_it(params.glove_embed)

########## Load dataset #############
dataset_object = wtwtDataset()
train_dataset = dataset_object.train_dataset
eval_dataset = dataset_object.eval_dataset
if params.dummy_run:
    eval_dataset = train_dataset
    target_names = []
else:
    eval_dataset = dataset_object.eval_dataset
    target_names = [dataset_object.id2stance[id_] for id_ in range(0, 4)]

########## Create model #############

class BERTStance(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.drop = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_stances)
        self.init_weights()
    def forward(self, text, att_mask, token_type):
        _, pooled = self.bert(text, attention_mask=att_mask, token_type_ids=token_type)
        return self.classifier(self.drop(pooled))

config = BertConfig.from_pretrained(params.bert_type)
config.num_stances = 4

model = BERTStance.from_pretrained(params.bert_type, config=config)

embedding_size = model.bert.embeddings.word_embeddings.weight.size(1)
new_embeddings = torch.FloatTensor(3, embedding_size).uniform_(-0.1, 0.1)
new_embedding_weight = torch.cat((model.bert.embeddings.word_embeddings.weight.data,new_embeddings), 0)
model.bert.embeddings.word_embeddings.weight.data = new_embedding_weight
print("Embedding Shape:", model.bert.embeddings.word_embeddings.weight.data.size())

model = model.to(params.device)
print("Detected", torch.cuda.device_count(), "GPUs!")
# model = torch.nn.DataParallel(model)

if params.wandb:
    wandb.watch(model)

########## Optimizer & Loss ###########

def my_fancy_optimizer(warmup_proportion=0.1):
    num_train_optimization_steps = len(train_dataset) * params.n_epochs

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    print(num_train_optimization_steps, warmup_proportion*num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr = params.lr, correct_bias=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                    num_warmup_steps = int(warmup_proportion*num_train_optimization_steps),
                    num_training_steps = num_train_optimization_steps)

    return optimizer, scheduler

# criterion = torch.nn.CrossEntropyLoss(weight=dataset_object.criterion_weights, reduction='sum')
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
# optimizer, scheduler = my_fancy_optimizer()
optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)

valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)
print(valid_loss)

for epoch in range(params.n_epochs):
    print("\n\n========= Beginning", epoch+1, "epoch ==========")

    train_loss = train(model, train_dataset, criterion)
    print("EVALUATING:")
    valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)

    if not params.dummy_run and params.wandb:
        wandb_dict = {}
        for labl in target_names:
            for metric, val in classify_report[labl].items():
                if metric != "support":
                    wandb_dict[labl + "_" + metric] = val

        wandb_dict["F1-Weighted"] = classify_report["weighted avg"]["f1-score"]
        wandb_dict["F1-Avg"] = classify_report["macro avg"]["f1-score"]

        wandb_dict["Accuracy"] = classify_report["accuracy"]

        wandb_dict["Train_loss"] = train_loss
        wandb_dict["Valid_loss"] = valid_loss

        wandb.log(wandb_dict)

    epoch_len = len(str(params.n_epochs))
    print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {valid_loss:.5f}')
    print(print_msg)

