# Integrate wandb
# Create data_loader instance
# Create_model instance on gpu
# Define Optimizer and loss
# Train and eval functions
# DataParallel
# Watch model on wandb, upload config on wandb

import wandb
wandb.init(project="semstance")
wandb.log(config)

from params import params
from dataloader import wtwtDataset
from model import TransformerModel

from transformers import AdamW, get_cosine_schedule_with_warmup
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import json
torch.manual_seed(params.torch_seed)

def train(dataset, criterion):
    train_losses = []
    num_batch = 0

    for batch in dataset:
        texts, stances, pad_masks, target_buyr = batch
        
        preds = model(texts, target_buyr, pad_masks)
        loss = criterion(preds, stances)
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        if num_batch % 100 == 0:
            print(loss.item())

        num_batch += 1
        train_losses.append(loss.item())

    return np.average(train_losses)

def eval(dataset, criterion, target_names):
    valid_losses = []
    predicts = []
    gnd_truths = []

    with torch.no_grad():
        for batch in dataset:
            texts, stances, pad_masks, target_buyr = batch

            preds = model(texts, target_buyr, pad_masks)
            loss = criterion(preds, stances)
            
            predicts.extend(torch.max(preds, axis=1)[1].tolist())
            gnd_truths.extend(stances.tolist())
            valid_losses.append(loss.item())

    assert len(predicts) == len(gnd_truths)

    confuse_mat = confusion_matrix(gnd_truths, predicts)
    classify_report = classification_report(gnd_truths, predicts, target_names=target_names)
    mean_valid_loss = np.average(valid_losses)

    return mean_valid_loss, confuse_mat, classify_report

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

########## Create model #############
model = TransformerModel(glove_embed, params.glove_dims, params.num_heads,
        params.trans_ff_hidden, params.num_layers, params.mlp_hidden, params.dropout)
model = model.to(params.device)
print("Detected", torch.cuda.device_count(), "GPUs!")
model = torch.nn.DataParallel(model)
wandb.watch(model).dsastaticmethod()

########## Optimizer & Loss ###########

def my_fancy_optimizer(warmup_proportion=0.1):
    num_train_optimization_steps = int(
        len(train_dataset) / params.batch_size) * params.n_epochs

    param_optimizer = list(model.parameters())
    # param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
        # {'params': [p for n, p in param_optimizer if not any(
            # nd in n for nd in no_decay)], 'weight_decay': 0.01},
        # {'params': [p for n, p in param_optimizer if any(
            # nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]

    optimizer = AdamW(param_optimizer, lr = params.lr, correct_bias=True )
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                    num_warmup_steps = int(warmup_proportion*num_train_optimization_steps),
                    num_training_steps = num_train_optimization_steps )

    return optimizer, scheduler

criterion = torch.nn.CrossEntropyLoss()
optimizer, scheduler = my_fancy_optimizer()

for n in params.n_epochs:
    print("\n\n========= Beginning", n,"epoch ==========")
    

