# import logging
import csv
from datautil import IndexedDataset, get_dataloader
from log import LOG_FILE_NAME, setup_logging, get_logger
import tqdm
import torch
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from model import Model

from utils import (
    encode_data,
    flatten_episodes,
    get_device,
    extract_episodes_from_json,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
)

# Get logger
setup_logging()
log = get_logger()

# Set up

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    train_eps, val_eps = extract_episodes_from_json(args.in_data_fn)
    train_data = flatten_episodes(train_eps)
    val_data = flatten_episodes(val_eps)

    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_eps, vocab_size=args.vocab_size)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_eps)

    with open("vocab.csv", "w") as f:
        for word, idx in vocab_to_index.items():
            f.write("%s, %d\n" % (word, idx))

    train_np_x, train_np_y = encode_data(train_data, vocab_to_index, actions_to_index, targets_to_index, len_cutoff)
    train_dataset = IndexedDataset([torch.from_numpy(xi) for xi in train_np_x], torch.from_numpy(train_np_y))

    val_np_x, val_np_y = encode_data(val_data, vocab_to_index, actions_to_index, targets_to_index, len_cutoff)
    val_dataset = IndexedDataset([torch.from_numpy(xi) for xi in val_np_x], torch.from_numpy(val_np_y))

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader, val_loader, (vocab_to_index, index_to_vocab, actions_to_index, index_to_actions, targets_to_index, index_to_targets)


def setup_model(args, device, vocab_to_index, actions_to_index, targets_to_index):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    model = Model(device,
            vocab_size=len(vocab_to_index),
            n_actions=len(actions_to_index),
            n_targets=len(targets_to_index),
            embedding_dim=args.emb_dim,
            lstm_hidden_dim=args.lstm_dim
        )
    return model


def setup_optimizer(args, model: torch.nn.Module):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.)

    return action_criterion, target_criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels, input_lengths) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs, input_lengths) # ,labels

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(actions_out.squeeze(), labels[:, 0].long())
        target_loss = target_criterion(targets_out.squeeze(), labels[:, 1].long())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    if training:
        return epoch_action_loss, epoch_target_loss, action_acc, target_acc
    else:
        return epoch_action_loss, epoch_target_loss, action_acc, target_acc, (action_preds, target_preds, action_labels, target_labels)


def validate(
    args, model, loader, maps, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc, outputs = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc, outputs


def train(args, model, loaders, maps, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    # maps
    vocab_to_index, index_to_vocab, actions_to_index, index_to_actions, targets_to_index, index_to_targets = maps

    # logging lists
    train_action_losses = []
    train_target_losses = []
    train_action_accs = []
    train_target_accs = []

    val_action_losses = []
    val_target_losses = []
    val_action_accs = []
    val_target_accs = []

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        log.info(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        log.info(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        train_action_losses.append(train_action_loss)
        train_target_losses.append(train_target_loss)
        train_action_accs.append(train_action_acc)
        train_target_accs.append(train_target_acc)

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc, (action_preds, target_preds, action_labels, target_labels) = validate(
                args,
                model,
                loaders["val"],
                maps,
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            model.train()

            log.info(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            log.info(
                f"val action acc : {val_action_acc} | val target acc: {val_target_acc}"
            )

            val_action_losses.append(val_action_loss)
            val_target_losses.append(val_target_loss)
            val_action_accs.append(val_action_acc)
            val_target_accs.append(val_target_acc)

            # show k random examples
            log_str = "Examples after %d epochs -\n" % epoch
            k_random_samples = np.random.choice(len(loaders["val"].dataset), size=10, replace=False)
            for idx in k_random_samples:
                log_str += "input :\t%s" % ' '.join([])
                action_preds

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #
    # plt.plot(train_action_losses)
    # plt.plot(val_action_losses)
    
    # plt.plot(train_action_accs)
    # plt.plot(val_action_accs)

    with open(LOG_FILE_NAME.split('.')[0] + "-train_action_losses.log", "w") as file:
        file.write('\n'.join(str(x) for x in train_action_losses))
    with open(LOG_FILE_NAME.split('.')[0] + "-train_action_accs.log", "w") as file:
        file.write('\n'.join(str(x) for x in train_action_accs))
    with open(LOG_FILE_NAME.split('.')[0] + "-val_action_losses.log", "w") as file:
        file.write('\n'.join(str(x) for x in val_action_losses))
    with open(LOG_FILE_NAME.split('.')[0] + "-val_action_accs.log", "w") as file:
        file.write('\n'.join(str(x) for x in val_action_accs))


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, device, maps[0], maps[2], maps[4])
    log.info(model)
    model.cuda()

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, maps, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn",       type=str,               help="data file")
    parser.add_argument("--model_output_dir", type=str,               help="where to save model outputs")
    parser.add_argument("--batch_size",       type=int, default=32,   help="size of each batch in loader")
    parser.add_argument("--num_epochs",       type=int, default=1000, help="number of training epochs")
    parser.add_argument("--val_every",                  default=5,    help="number of epochs between every eval loop")
    
    parser.add_argument("--force_cpu",        action="store_true",    help="debug mode")
    parser.add_argument("--eval",             action="store_true",    help="run eval")

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument("--vocab_size",    type=int,   default=10000, help="max size of vocabulary")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate (alpha)")
    parser.add_argument("--emb_dim",       type=int,   required=True, help="embedding dimension")
    parser.add_argument("--lstm_dim",      type=int,   required=True, help="lstm hidden state dimension")

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        log.exception(e)
        log.handlers[1].flush()
