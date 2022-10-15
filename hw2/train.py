import argparse
import os
import tqdm
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from eval_utils import downstream_validation
from log import get_logger, next_path, setup_logging
from model import CBOWModel, SkipGramModel
import utils
import data_utils
import dataloaders
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

setup_logging()
log = get_logger()

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # ================== TODO: CODE HERE ================== #
    # Task: Given the tokenized and encoded text, you need to
    # create inputs to the LM model you want to train.
    # E.g., could be target word in -> context out or
    # context in -> target word out.
    # You can build up that input/output table across all
    # encoded sentences in the dataset!
    # Then, split the data into train set and validation set
    # (you can use utils functions) and create respective
    # dataloaders.
    # ===================================================== #
    len_sentences = len(encoded_sentences)
    train_split_ratio = 0.8
    train_data_split = int(train_split_ratio * len_sentences)
    # val_data_split = len_sentences - train_data_split

    if args.model == "cbow":
        train_dataset = dataloaders.create_dataset_cbow(encoded_sentences[:train_data_split], lens[:train_data_split], 4, pad=not args.no_pad)
        val_dataset = dataloaders.create_dataset_cbow(encoded_sentences[train_data_split:], lens[train_data_split:], 4, pad=not args.no_pad)
        # train_loader = dataloaders.get_dataloader(train_dataset, args.model, args.context_len, args.batch_size, shuffle=True)
        # val_loader = dataloaders.get_dataloader(val_dataset, args.model, args.context_len, args.batch_size, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_dataset = dataloaders.create_dataset_skipgram(encoded_sentences[:train_data_split], lens[:train_data_split], 4, pad=not args.no_pad)
        val_dataset = dataloaders.create_dataset_skipgram(encoded_sentences[train_data_split:], lens[train_data_split:], 4, pad=not args.no_pad)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader, vocab_to_index, index_to_vocab


def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #
    if args.model == "cbow":
        model = CBOWModel(args.vocab_size, args.embedding_dim)
    elif args.model == "skipgram":
        model = SkipGramModel(args.vocab_size, args.embedding_dim)
    return model


def setup_optimizer(args, model: torch.nn.Module):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for predictions. 
    # Also initialize your optimizer.
    # ===================================================== #
    if args.model == "cbow":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.model == "skipgram":
        criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        # put model inputs to device
        inputs, labels = inputs.to(device).long(), labels.to(device).long()

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs)

        # calculate prediction loss
        if args.model == "cbow":
            loss = criterion(pred_logits.squeeze(), labels)
        elif args.model == "skipgram":
            labels_mhe = dataloaders.create_multihot_from_labels(labels, model.vocab_size, device)
            if not args.no_pad:
                labels_mhe[:, model.v2i['<pad>']] = 0.
            loss = criterion(pred_logits.squeeze(), labels_mhe)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        if args.model == "cbow":
            preds = pred_logits.argmax(-1)
            pred_labels.extend(preds.cpu().numpy())
            target_labels.extend(labels.cpu().numpy())
        elif args.model == "skipgram":
            _, preds = torch.topk(pred_logits, args.context_len * 2, dim=1)
            pred_labels.extend(map(set, preds.cpu().numpy()))
            target_labels.extend(map(set,labels.cpu().numpy()))
            # pred_labels.extend(preds.cpu().numpy())
            # target_labels.extend(labels.cpu().numpy())

    if args.model == "cbow":
        acc = accuracy_score(pred_labels, target_labels)
    elif args.model == "skipgram":
        # pred_labels = [set(p) for p in pred_labels]
        # target_labels = [set(t) for t in target_labels]
        # I = np.array([len(p.intersection(l)) for p, l in zip(pred_labels, target_labels)])
        # U = np.array([len(p.union(l)) for p,l in zip(pred_labels, target_labels)])
        IoU = np.array([len(p.intersection(l)) / len(p.union(l)) for p, l in zip(pred_labels, target_labels)])
        # IoU = [i / u for i,u in zip(I, U)]
        # IoU = I / U
        acc = IoU.mean()

    epoch_loss /= len(loader)

    return epoch_loss, acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def main(args):
    device = utils.get_device(args.force_cpu)

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, v2i, i2v = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args)
    setattr(model, "v2i", v2i)
    setattr(model, "i2v", i2v)
    model.to(device)
    log.info(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    num_train = len(train_loader.dataset)
    num_val = len(val_loader.dataset)

    train_losses = []
    train_accs = []

    val_losses = []
    val_accs = []

    for epoch in range(args.num_epochs):
        # train model for a single epoch
        log.info(f"Epoch {epoch}")
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )
        log.info(f"train loss : {train_loss} | train acc: {train_acc}")

        train_losses.append(train_loss / num_train)
        train_accs.append(train_acc)

        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )
            log.info(f"val loss : {val_loss} | val acc: {val_acc}")
            
            val_losses.append(val_loss / num_val)
            val_accs.append(val_acc)

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #

            # save word vectors
            word_vec_file = os.path.join(args.output_dir, args.word_vector_fn)
            log.info("saving word vec to %s", word_vec_file)
            data_utils.save_word2vec_format(word_vec_file, model, i2v)

            # evaluate learned embeddings on a downstream task
            downstream_validation(word_vec_file, external_val_analogies)


        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.output_dir, "model.ckpt")
            log.info("saving model to %s", ckpt_file)
            torch.save(model, ckpt_file)

    plots_filepath = next_path("logs/run-%s-plots.png")

    fig, axs = plt.subplots(1, 2, sharex=True)

    axs[0].plot(np.arange(0, args.num_epochs, 1), train_accs, label="train")
    axs[0].plot(np.arange(0, args.num_epochs, args.val_every), val_accs, label="val")
    axs[0].set_title("Accuracy")
    # plt.savefig(store_file_prefix + "-action_accuracies.png")

    axs[1].plot(np.arange(0, args.num_epochs, 1), train_losses, label="train")
    axs[1].plot(np.arange(0, args.num_epochs, args.val_every), val_losses, label="val")
    axs[1].set_title("Loss")
    # plt.savefig(store_file_prefix + "-action_losses.png")

    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')

    fig.savefig(plots_filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="where to save training outputs", default="outputs")
    parser.add_argument("--data_dir", type=str, help="where the book dataset is stored", required=True)
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there 
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, help="filepath to the analogies json file",
        default="analogies_v3000_1309.json"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=30, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=5,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="learning rate"
    )
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument(
        "--model",
        choices=["cbow", "skipgram"],
        help="type of model to use - CBOW or Skip-Gram",
        required=True,
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        help="dimensions of the embedding vector for each word",
        default=300,
    )
    parser.add_argument(
        "--context_len",
        type=int,
        help="length of the context to use",
        default=4,
    )
    parser.add_argument(
        "--no_pad",
        action="store_true",
        help="do not use padded sequences (includes corner elements, increases dataset size)"
    )

    args = parser.parse_args()
    log.info("Args:\n\t%s", "\n\t".join(f"{k} = {v}" for k, v in vars(args).items()))
    main(args)
