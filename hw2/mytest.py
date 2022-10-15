from log import get_logger, setup_logging
import data_utils
import dataloaders
import numpy as np

setup_logging()
log = get_logger()

class Args(object):
    pass

args = Args()
setattr(args, 'data_dir', 'books')
setattr(args, 'vocab_size', 3000)
setattr(args, 'batch_size', 32)
setattr(args, 'analogies_fn', 'analogies_v3000_1309.json')
setattr(args, 'word_vector_fn', 'learned_word_vectors.txt')
setattr(args, 'num_epochs', 10)
setattr(args, 'model', 'cbow')

sentences = data_utils.process_book_dir(args.data_dir)

(
    vocab_to_index,
    index_to_vocab,
    suggested_padding_len,
) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

encoded_sentences, lens = data_utils.encode_data(
    sentences,
    vocab_to_index,
    suggested_padding_len,
)

dataset = dataloaders.create_dataset_cbow(encoded_sentences, lens, 4)


# dataloader = dataloaders.get_dataloader(dataset, args.model, 4, args.batch_size, shuffle=True)

# for (i, o) in dataloader:
#     break