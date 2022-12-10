from pathlib import Path

import torch
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer

from models import Seq2Seq
from train import MAX_LEN
from perturbations.apply_perturbations import apply_perturbation_to_text
from text import Text
import argparse

# gets the predicted output for a given input sequence by using our model
def predict(transformed_text, model, tokenizer, classification_token_id, separator_token_id, maximum_length=MAX_LENGTH):
    word = tokenizer.encode(transformed_text).ids
    word = torch.tensor(word, dtype=torch.long)
    word = word.unsqueeze(0)
    memory = model.encode_word(word)
    transformed_word = torch.zeros((word.shape[0], maximum_length), dtype=torch.long)
    transformed_word[:, 0] = classification_token_id

    for i in range(1, maximum_length):
        output = model.decode_transformed_word(transformed_word[:, :i], memory=memory)
        output = output.argmax(2)
        is_end = output == separator_token_id
        is_end, _ = is_end.max(1)
        if is_end.sum() == output.shape[0]:
            break
        transformed_word[:, i] = output[:, -1]

    return tokenizer.decode(transformed_word.squeeze().numpy())


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
    args = argument_parser.parse_args()
    
    # create the Seq2Seq model to use to get the predicted output
    model = Seq2Seq(
        output_vocab_size=Tokenizer.from_file(args.tokenizer_location).get_vocab_size(),
        padding=Tokenizer.from_file(args.tokenizer_location).token_to_id("[PAD]"),
    )
    # evaluate the performance of the model
    device = torch.device("cpu")
    model.load_state_dict(torch.load(args.model_location, map_path=device)["state_dict"])
    model.eval()
    
    # continuously get input from the user and apply the model to this input and then output the predicted autocorrected sequence of text
    while True:
        s = input("\nInput text:\n> ")
        predicted_output = predict(
            transformed_text=s,
            model=model,
            tokenizer=tokenizer,
            classification_token_id=tokenizer.token_to_id("[CLS]"),
            separator_token_id=tokenizer.token_to_id("[SEP]"),
        )
        print("Output text: ")
        print("> " + predicted_output)