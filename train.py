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
from dataloader import wtwtDataset
from model_lstm import LSTMModel

from transformers import AdamW, get_cosine_schedule_with_warmup
import torch
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
        (texts, stances, pad_masks, target_buyr, edge_labels, lstm_root_masks, lstm_child_masks,
            gatt_masks_for_root, gatt_root_idxs, semantics_root_mask) = batch
        preds = model(texts, target_buyr, pad_masks, edge_labels, lstm_root_masks,
                            lstm_child_masks, gatt_masks_for_root, gatt_root_idxs, semantics_root_mask)
        loss = criterion(preds, stances)
        
        loss.backward()
        optimizer.step()
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
            (texts, stances, pad_masks, target_buyr, edge_labels, lstm_root_masks, lstm_child_masks,
                    gatt_masks_for_root, gatt_root_idxs, semantics_root_mask) = batch
            preds = model(texts, target_buyr, pad_masks, edge_labels, lstm_root_masks,
                    lstm_child_masks, gatt_masks_for_root, gatt_root_idxs, semantics_root_mask)
            
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

model = LSTMModel(glove_embed, params.glove_dims,
                params.fusion_alpha, dataset_object.num_edge_labels,
                params.message_passing_hidden,
                params.num_gatt_layers, params.gatt_dropout, 
                classifier_mlp_hidden=params.mlp_hidden,
                bidirectional=True,
                dropout=params.dropout)

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
    # print(optimizer_grouped_parameters)
    print(num_train_optimization_steps, warmup_proportion*num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr = params.lr, correct_bias=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                    num_warmup_steps = int(warmup_proportion*num_train_optimization_steps),
                    num_training_steps = num_train_optimization_steps)

    return optimizer, scheduler

criterion = torch.nn.CrossEntropyLoss(weight=dataset_object.criterion_weights, reduction='sum')
# criterion = torch.nn.CrossEntropyLoss()
# optimizer, scheduler = my_fancy_optimizer()
optimizer = torch.optim.SGD(model.parameters(), lr = params.lr)

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

