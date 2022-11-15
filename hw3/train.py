import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from log import setup_logging, get_logger

from utils import (
    get_device,
    extract_episodes_from_json,
    flatten_episodes,
    encode_data,
    IndexedDataset,
    get_dataloader,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    prefix_match
)

from model import Encoder, Decoder, EncoderDecoder

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
    train_eps = train_eps[:1000]
    train_data = flatten_episodes(train_eps)
    val_data = flatten_episodes(val_eps)

    vocab_to_index, index_to_vocab, _ = build_tokenizer_table(train_eps, vocab_size=args.vocab_size)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_eps)

    train_np_x, train_np_y = encode_data(train_data, vocab_to_index, actions_to_index, targets_to_index)
    val_np_x, val_np_y = encode_data(val_data, vocab_to_index, actions_to_index, targets_to_index)

    train_dataset = IndexedDataset([torch.from_numpy(xi) for xi in train_np_x], [torch.from_numpy(yi) for yi in train_np_y])
    val_dataset = IndexedDataset([torch.from_numpy(xi) for xi in val_np_x], [torch.from_numpy(yi) for yi in val_np_y])

    train_loader = get_dataloader(train_dataset, args.batch_size)
    val_loader = get_dataloader(val_dataset, args.batch_size)
    return train_loader, val_loader, (vocab_to_index, index_to_vocab, actions_to_index, index_to_actions, targets_to_index, index_to_targets)


def setup_model(args, device, maps):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #
    enc = Encoder(device, args.emb_dim, args.hidden_dim, maps[0])
    dec = Decoder(device, args.emb_dim, args.hidden_dim, maps[2], maps[4])
    model = EncoderDecoder(device, enc, dec, args.hidden_dim, maps[2], maps[4])
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    criterion = (torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss())
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.1)
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
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_loss = 0.0
    epoch_acc = 0.0

    done = 0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, x_lens, labels, y_lens) in loader:
        print("next minibatch : ", done)
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        out_actions, out_targets = model(inputs, x_lens, labels, y_lens)

        loss = 0.0
        if not args.student_forcing:
            print("labels size : ", labels.size())
            for oidx in range(labels.size(1)):
                loss += criterion[0](out_actions[:,oidx,:], labels[:,oidx,0].long())
                loss += criterion[1](out_targets[:,oidx,:], labels[:,oidx,1].long())

        print("loss : %s" % loss.item())
        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            print("completed backward")
            optimizer.step()
            done += len(inputs)
            print("completed optimizer step : %s" % done)

        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        if not training:
            # TODO: add code to log these metrics
            output = torch.cat((torch.argmax(out_actions, dim=2, keepdim=True), torch.argmax(out_actions, dim=2, keepdim=True)), dim=2)
            em = output == labels
            prefix_em = prefix_match(output, labels)
            acc = 0.0

            # logging
            epoch_acc += acc.item()

        epoch_loss += loss.item()

    log.debug("calculating epoch loss...")
    epoch_loss /= len(loader)
    epoch_acc /= len(loader)

    return epoch_loss, epoch_acc


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


def train(args, model, loaders, optimizer, criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        # some logging
        log.info(f"train loss : {train_loss}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
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

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, device, maps)
    log.info(model)
    model.to(device)

    # get optimizer and loss functions
    criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument("--vocab_size", default=1000, help="size of input vocabulary")
    parser.add_argument("--emb_dim", default=100, help="embedding dimension")
    parser.add_argument("--hidden_dim", default=100, help="hidden dimension for LSTMs")
    parser.add_argument("--learning_rate", default=0.1, help="learning rate")
    parser.add_argument("--student_forcing", action="store_true", help="use student forcing during training")

    args = parser.parse_args()

    main(args)
