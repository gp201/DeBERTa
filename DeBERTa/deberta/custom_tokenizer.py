import sentencepiece as sp
import six
import unicodedata
import os
import regex as re
from ..utils import get_logger
logger=get_logger()


import pdb

__all__ = ['CustomTokenizer']

class CustomTokenizer:
    def __init__(self, vocab_file, do_lower_case=False, special_tokens=None, bpe_dropout=0, split_by_punct=False):
        self.split_by_punct = split_by_punct
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.special_tokens = []
        if special_tokens is not None:
            self.special_tokens.extend(special_tokens)
        for t in ['[PAD]', '[CLS]', '[SEP]', '[UNK]', '[MASK]']:
            self.add_special_token(t)
        logger.info(f'Vocab: {self.vocab}')

    def add_special_token(self, token):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)
            self.ids_to_tokens[self.vocab[token]] = token
            self.special_tokens.append(token)

    def load_vocab(self, vocab_file):
        # read the vocab file
        with open(vocab_file, 'r') as f:
            lines = f.readlines()
            # remove empty lines
            lines = [line.strip() for line in lines if line.strip()]

        # create a dictionary to store the vocab
        vocab = {}
        # iterate over the lines
        # for line in lines:
        #     # split the line by the tab character
        #     parts = line.split('\t')
        #     # get the token and the id
        #     token, id = parts[0], int(parts[1])
        #     # add the token to the vocab
        #     vocab[token] = id
        for i, line in enumerate(lines):
            vocab[line] = i

        return vocab

    def tokenize(self, text, max_seq_length=512):
        # clean the text
        text = self._clean_text(text)
        # add bos token
        # tokens = [self.bos()]
        tokens = []
        tokens.extend([char if char in self.vocab else self.unk() for char in text])
        # pad the tokens
        if len(tokens) < max_seq_length:
            tokens.extend([self.pad()] * (max_seq_length - len(tokens) - 2))
        # add eos token
        # tokens.append(self.eos())
        return tokens

    def _clean_text(self, text):
      """Performs invalid character removal and whitespace cleanup on text."""
      output = []
      for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or self._is_control(char):
          continue
        if self._is_whitespace(char):
            continue
        else:
          output.append(char)
      return "".join(output)

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically contorl characters but we treat them
        # as whitespace since they are generally considered as such.
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        # These are technically control characters but we count them as whitespace
        # characters.
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[t] if t in self.vocab else 1 for t in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token[id] for id in ids if id in self.id_to_token]

    def pad(self):
        return '[PAD]'

    def bos(self):
        return '[CLS]'

    def eos(self):
        return '[SEP]'

    def unk(self):
        return '[UNK]'

    def mask(self):
        return '[MASK]'

    def sym(self, id):
        return self.ids_to_tokens[id]

