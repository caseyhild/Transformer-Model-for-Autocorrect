import random
from functools import partial
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from models import Seq2Seq
from perturbations.apply_perturbations import apply_perturbation_to_text
from text import Text
import argparse

MAX_LENGTH = 256

def get_data_batch(data, padding):
    # get the list of words and the corresponding list of transformed words for each word
    transformed_words = []
    words = []
    for (word, transformed_word) in data:
        transformed_words.append(transformed_word)
        words.append(word)
    # add padding between both original words and transformed words
    transformed_words = pad_sequence(transformed_words, padding_value=padding, batch_first=True)
    words = pad_sequence(words, padding_value=padding, batch_first=True)
    return words, transformed_words

class Data(torch.utils.data.Dataset):
    def get_data(self, _):
        # get data point at a random index (one sentence)
        data = self.data[random.randint(0, self.length - 1)]
        text_string = Text(original=data)

        apply_perturbation_to_text(text)

        # encode both original and transformed text
        transformed_text_encoded = self.tokenizer.encode(text_string.transformed).ids
        original_text_encoded = self.tokenizer.encode(data).ids

        # first MAX_LENGTH characters of encoded transformed text
        transformed_text_encoded = transformed_text_encoded[:MAX_LENGTH]
        # first MAX_LENGTH characters of encoded original text
        original_text_encoded = original_text_encoded[:MAX_LENGTH]

        # pass both encoded original and transformed text to torch.tensor
        transformed_text_encoded = torch.tensor(transformed_text_encoded, dtype=torch.long)
        original_text_encoded = torch.tensor(original_text_encoded, dtype=torch.long)

        return transformed_text_encoded, original_text_encoded
    
    def get_length(self):
        # get the total size of the data set
        return self.length
    
    def __init__(self, data, tokenizer):
        # initialize data and tokenizer
        self.data = data
        self.length = len(self.data)
        self.tokenizer = tokenizer


if __name__ == "__main__":
    # create arguments for locations/paths of base, tokenizer, data, and model
    argument_parser = argparse.ArgumentParser()    
    argument_parser.add_argument(
        "--base_location", default=str(Path(__file__).absolute().parents[0]/"output")
    )
    argument_parser.add_argument(
        "--tokenizer_location",
        default=str(
            Path(__file__).absolute().parents[0]/"resources"/"tokenizer.json"
        ),
    )
    argument_parser.add_argument(
        "--data_location", default="wikisent2.txt" # read in data from text file of wikipedia sentences
    )
    argument_parser.add_argument(
        "--model_location",
        default=str(
            Path(__file__).absolute().parents[0]/"output"/"checker.ckpt"
        ),
    )
    argument_parser.add_argument("--size", default=32)
    argument_parser.add_argument("--iterations", default=2000)
    args = argument_parser.parse_args()

    # make the directory for the base location
    base_location = Path(args.base_location)
    base_location.mkdir(exist_ok=True)
    
    # read all of the data from the file line by line
    data_location = args.data_location
    with open(data_location) as f:
        data = f.read().split("\n")

    # split the data into two sets: training set and test set
    training_set, test_set = train_test_split(data, test_size=0.20, random_state=1337)

    training_data = Data(data=training_set, tokenizer=Tokenizer.from_file(args.tokenizer_location))
    test_data = Data(data=test_set, tokenizer=Tokenizer.from_file(args.tokenizer_location))
    
    # load all of the training data and test data into database
    training_data_loader = DataLoader(
        training_data,
        batch_size=args.size,
        num_workers=10,
        shuffle=True,
        collate_fn=partial(get_data_batch, padding=Tokenizer.from_file(args.tokenizer_location).token_to_id("[PAD]")),
    )
    test_data_loader = DataLoader(
        test_data,
        batch_size=args.size,
        num_workers=10,
        shuffle=True,
        collate_fn=partial(get_data_batch, padding=Tokenizer.from_file(args.tokenizer_location).token_to_id("[PAD]")),
    )

    # create the Seq2Seq model to train with this dataset
    model = Seq2Seq(
        output_vocab_size=Tokenizer.from_file(args.tokenizer_location).get_vocab_size(),
        padding=Tokenizer.from_file(args.tokenizer_location).token_to_id("[PAD]"),
    )

    # set a checkpoint and log the tensor board
    checkpoint_callback = ModelCheckpoint(
        monitor="valid_access", mode="max", dirpath=base_location, filename="checker"
    )
    
    logger = TensorBoardLogger(
        save_dir=str(base_location),
        name="logs",
    )

    # train the model on the dataset
    pl.Trainer(
        max_epochs=args.iterations,
        accelerator='cpu',
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    ).fit(model, training_data_loader, test_data_loader)
