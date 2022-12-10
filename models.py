import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import functional as F

def get_transformed_word_mask(length, device):
    # creates a triangular matrix filled with negative infinity and 1 on the diagonal
    return torch.triu(torch.ones(length, length, device=device) * float("-inf"), diagonal=1)

def add_padding(tensor, padding):
    # if entry is equal to padding, sets it to 1, 0 otherwise
    return (tensor == padding).transpose(0, 1)

def get_accuracy(y: torch.Tensor, y_predicted: torch.Tensor, padding):
    # returns the accuracy of a given output based on y and y_predicted
    y = torch.masked_select(y, y != padding)
    y_predicted = torch.masked_select(y_predicted, y != padding)
    return (y == y_predicted).double().mean()

# encodes the given tensor and applies dropout to the result
class PositionalEncoding(nn.Module):
    # aplies dropout to the positional encoding of a given tensor
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x += self.positional_encoding[: x.size(0)]
        return self.dropout(x)
    
    # initializes the positional encoding
    def __init__(self, channels: int, dropout: float = 0.1, maximum_length: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(maximum_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2) * (-math.log(10000.0) / channels))
        positional_encoding = torch.zeros(maximum_length, 1, channels)
        positional_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", positional_encoding)

# embedds each of the given tokens
class TokenEmbedding(nn.Module):
    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_length)
    
    # initializes the token embedding
    def __init__(self, vocab_size: int, embedding_length):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_length)
        self.embedding_length = embedding_length

# the model to transform a given sequence into another sequence
class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        output_vocab_size,
        padding,
        channels=256,
        dropout=0.1,
        learning_rate=1e-6,
    ):
        super().__init__()

        self.output_vocab_size = output_vocab_size
        self.padding = padding
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.embeddings = TokenEmbedding(vocab_size=self.output_vocab_size, embedding_length=channels)
        self.positional_encoding = PositionalEncoding(channels=channels, dropout=dropout)
        self.transformer = torch.nn.Transformer(
            channels=channels,
            nhead=4,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=dropout,
        )
        self.linear = Linear(channels, output_vocab_size)
        self.do = nn.Dropout(p=self.dropout)

    # initializes the weights
    def init_weights(self) -> None:
        init_range = 0.1
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    #encodes a given word
    def encode_word(self, word):
        word = word.permute(1, 0)
        word_pad_mask = create_padding_mask(word, self.pad_idx)
        word = self.embeddings(word)
        word = self.pos_encoder(word)
        word = self.transformer.encoder(word, word_key_padding_mask=word_pad_mask)
        return self.pos_encoder(word)
    
    # decodes a given word that has been transformed
    def decode_transformed_word(self, transformed_word, memory):
        transformed_word = transformed_word.permute(1, 0)
        output_sequence_length = transformed_word.size(0)
        size = transformed_word.size(1)
        transformed_word_with_padding = add_padding(transformed_word, self.padding)
        transformed_word = self.pos_encoder(self.embeddings(transformed_word))
        transformed_word_mask = get_transformed_word_mask(out_sequence_len, transformed_word.device)
        return self.linear(self.transformer.decoder(tgt=transformed_word, memory=memory, tgt_mask=transformed_word_mask,
                                       tgt_key_padding_mask=transformed_word_with_padding).permute(1, 0, 2))

    # perform forward-propogation through our model
    def forward(self, x):
        word, transformed_word = x
        return self.decode_transformed_word(transformed_word=transformed_word, memory=self.encode_word(word))

    def training_step(self, data, index):
        return self._step(data, index, name="train")

    def validation_step(self, data, index):
        return self._step(data, index, name="valid")

    def test_step(self, data, index):
        return self._step(data, index, name="test")

    # applies one step through our model, calculating hte predicted y and calculating the loss of the model
    def _step(self, data, index, name="train"):
        word, transformed_word = data
        transformed_word_input = transformed_word[:, :-1]
        transformed_word_output = transformed_word[:, 1:]
        y_predicted = self((word, transformed_word_input))
        y_predicted = y_predicted.view(-1, y_predicted.size(2))
        y = transformed_word_output.contiguous().view(-1)
        loss = F.cross_entropy(y_predicted, y, ignore_index=self.padding)
        _, predicted = torch.max(y_predicted, 1)
        accuracy = get_accuracy(y, predicted, padding=self.padding)
        self.log(f"{name}_loss", loss)
        self.log(f"{name}_accuracy", accuracy)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    n = 100
    # generate a random word
    word = torch.randint(low=0, high=n, size=(20, 16))
    transformed_word = torch.randint(low=0, high=n, size=(20, 32))
    # create the Seq2Seq model
    text_transformer = Seq2Seq(output_vocab_size=n, padding=0)
    # get the result from a given word and its transformation
    output = text_transformer((word, transformed_word))
    # print out the results
    print(output.size())
    print(output)
