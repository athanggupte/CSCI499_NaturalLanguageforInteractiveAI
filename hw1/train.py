# import logging
from cProfile import label
import csv
from datautil import IndexedDataset, get_dataloader
from log import log_filename, next_path, setup_logging, get_logger, last_path
import tqdm
import torch
import torchmetrics
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from model import ActionToTargetModel, AttentionModel, Model, TargetToActionModel

from utils import (
    encode_data,
    flatten_episodes,
    get_device,
    extract_episodes_from_json,
    log_examples,
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
    train_data, len_cutoff = flatten_episodes(train_eps, args.context)
    val_data, _ = flatten_episodes(val_eps, args.context)

    vocab_to_index, index_to_vocab = build_tokenizer_table(train_eps, vocab_size=args.vocab_size)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_eps)

    # with open("vocab.csv", "w") as f:
    #     for word, idx in vocab_to_index.items():
    #         f.write("%s, %d\n" % (word, idx))

    if args.model_type in ["lstm", "act-tar", "tar-act"]:
        len_cutoff = 0 if args.no_len_limit else len_cutoff
    # else:
    #     len_cutoff = max_len if args.no_len_limit else len_cutoff

    train_np_x, train_np_y = encode_data(train_data, vocab_to_index, actions_to_index, targets_to_index, len_cutoff)
    train_dataset = IndexedDataset([torch.from_numpy(xi) for xi in train_np_x], torch.from_numpy(train_np_y))

    val_np_x, val_np_y = encode_data(val_data, vocab_to_index, actions_to_index, targets_to_index, len_cutoff)
    val_dataset = IndexedDataset([torch.from_numpy(xi) for xi in val_np_x], torch.from_numpy(val_np_y))

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader, (vocab_to_index, index_to_vocab, actions_to_index, index_to_actions, targets_to_index, index_to_targets), len_cutoff


def setup_model(args, device, max_len, vocab_to_index, actions_to_index, targets_to_index):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    if args.model_type == "lstm":
        model = Model(device, max_len,
                vocab_size=len(vocab_to_index),
                n_actions=len(actions_to_index),
                n_targets=len(targets_to_index),
                embedding_dim=args.emb_dim,
                lstm_hidden_dim=args.lstm_dim
            )
    elif args.model_type == "attn":
        model = AttentionModel(device, max_len,
                vocab_size=len(vocab_to_index),
                n_actions=len(actions_to_index),
                n_targets=len(targets_to_index),
                embedding_dim=args.emb_dim,
                lstm_hidden_dim=args.lstm_dim
            )
    elif args.model_type == "act-tar":
        model = ActionToTargetModel(device, max_len,
                vocab_size=len(vocab_to_index),
                n_actions=len(actions_to_index),
                n_targets=len(targets_to_index),
                embedding_dim=args.emb_dim,
                lstm_hidden_dim=args.lstm_dim
            )
    elif args.model_type == "tar-act":
        model = TargetToActionModel(device, max_len,
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
    
    # keep track of inputs, preds and labels for evaluation (only for validation)
    val_inputs = []

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
        val_inputs.extend(inputs[:].cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    if training:
        return epoch_action_loss, epoch_target_loss, action_acc, target_acc
    else:
        return epoch_action_loss, epoch_target_loss, action_acc, target_acc, (action_preds, target_preds, action_labels, target_labels, val_inputs)


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

    model_filepath = next_path("models/model-%s.pt")

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
        if epoch % args.val_every == 0 or epoch == args.num_epochs - 1:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc, (action_preds, target_preds, action_labels, target_labels, inputs) = validate(
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
            log.info("Examples after %d epochs -\n" % epoch)
            log_examples(
                inputs,
                action_preds, action_labels,
                target_preds, target_labels,
                index_to_vocab, index_to_actions, index_to_targets,
                args.num_examples
            )
        else:
            val_action_losses.append(val_action_losses[-1])
            val_target_losses.append(val_target_losses[-1])
            val_action_accs.append(val_action_accs[-1])
            val_target_accs.append(val_target_accs[-1])

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #

    store_file_prefix = "logs/" + model_filepath.split('/')[1].split('.')[0]

    fig, axs = plt.subplots(2, 2, sharex=True)

    axs[0,0].plot(train_action_accs, label="train")
    axs[0,0].plot(val_action_accs, label="val")
    axs[0,0].set_title("Action accuracy")
    axs[0,0].set_xlabel
    # plt.savefig(store_file_prefix + "-action_accuracies.png")

    axs[0,1].plot(train_target_accs, label="train")
    axs[0,1].plot(val_target_accs, label="val")
    axs[0,1].set_title("Target accuracy")
    # plt.savefig(store_file_prefix + "-target_accuracies.png")

    axs[1,0].plot([l / len(loaders['train'].dataset) for l in train_action_losses], label="train")
    axs[1,0].plot([l / len(loaders['val'].dataset) for l in val_action_losses], label="val")
    axs[1,0].set_title("Action Loss")
    # plt.savefig(store_file_prefix + "-action_losses.png")
    
    axs[1,1].plot([l / len(loaders['train'].dataset) for l in train_target_losses], label="train")
    axs[1,1].plot([l / len(loaders['val'].dataset) for l in val_target_losses], label="val")
    axs[1,1].set_title("Target Loss")
    # plt.savefig(store_file_prefix + "-target_losses.png")

    handles, labels = axs[1,1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')

    fig.savefig(store_file_prefix + "-plots.png")

    torch.save(model.state_dict(), model_filepath)


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps, max_len = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, device, max_len, maps[0], maps[2], maps[4])
    log.info(model)
    model.to(device)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        # model_filepath = "models/model-%s.pt" % args.eval_model_id if args.eval_model_id != -1 else last_path("models/model-%s.pt")
        model_filepath = args.eval_model_path or last_path("models/model-%s.pt")
        model.load_state_dict(torch.load(model_filepath))

        val_action_loss, val_target_loss, val_action_acc, val_target_acc, (action_preds, target_preds, action_labels, target_labels, inputs) = validate(
            args,
            model,
            loaders["val"],
            maps,
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        log.info(
            f"action loss : {val_action_loss} | target loss: {val_target_loss}"
        )
        log.info(
            f"action acc : {val_action_acc} | target acc: {val_target_acc}"
        )

        log.info("Examples:")
        log_examples(
            inputs,
            action_preds, action_labels,
            target_preds, target_labels,
            maps[1], maps[3], maps[5],
            args.num_examples
        )

        action_precision, action_recall = torchmetrics.functional.precision_recall(torch.tensor(action_preds), torch.tensor(action_labels), average="micro")
        target_precision, target_recall = torchmetrics.functional.precision_recall(torch.tensor(target_preds), torch.tensor(target_labels), average="micro")

        action_f1score = 2 * action_precision * action_recall / (action_precision + action_recall)
        target_f1score = 2 * target_precision * target_recall / (target_precision + target_recall)

        log.info("Actions precision : %0.3f | recall : %0.3f | f1-score : %0.3f", action_precision, action_recall, action_f1score)
        log.info("Targets precision : %0.3f | recall : %0.3f | f1-score : %0.3f", target_precision, target_recall, target_f1score)

    else:
        train(
            args, model, loaders, maps, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn",       type=str, required=True, help="data file")
    parser.add_argument("--model_output_dir", type=str,                help="where to save model outputs")
    parser.add_argument("--batch_size",       type=int, default=32,    help="size of each batch in loader")
    parser.add_argument("--num_epochs",       type=int, default=1000,  help="number of training epochs")
    parser.add_argument("--val_every",        type=int, default=5,     help="number of epochs between every eval loop")
    
    parser.add_argument("--force_cpu",        action="store_true",     help="debug mode")
    parser.add_argument("--eval",             action="store_true",     help="run eval")
    # parser.add_argument("--eval_model_id",    type=int, default=-1,    help="id of model to evaluate (default: -1, last saved model)")
    parser.add_argument("--eval_model_path",  type=str,                help="filepath of the model to evaluate")
    parser.add_argument("--num_examples",     type=int, default=10,    help="number of random examples to print")

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument("--vocab_size",    type=int,   default=1000,  help="max size of vocabulary")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate (alpha)")
    parser.add_argument("--emb_dim",       type=int,   required=True, help="embedding dimension")
    parser.add_argument("--lstm_dim",      type=int,   required=True, help="lstm hidden state dimension")
    parser.add_argument("--no_len_limit",  action="store_true",       help="don't limit length of inputs")
    
    parser.add_argument("--model_type",    choices=["lstm", "attn", "act-tar", "tar-act"], default="lstm", help="specify the model architecture to use")
    parser.add_argument("--context",       choices=["curr", "prev", "next", "prev-next"], default="curr", help="specify the context to use")

    args = parser.parse_args()

    try:
        log.info("Args:\n\t%s", "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))

        main(args)
    except Exception as e:
        log.exception(e)
        log.handlers[1].flush()
