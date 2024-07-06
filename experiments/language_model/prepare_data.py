# coding: utf-8

import sys
sys.path.append('../../')

from DeBERTa import deberta
import sys
import argparse
from tqdm import tqdm

def tokenize_data(input, output=None, max_seq_length=512, vocab_path=None, vocab_type='custom'):
  p,t=deberta.load_vocab(vocab_path=vocab_path, vocab_type=vocab_type, pretrained_id='deberta-v3-base')
  tokenizer=deberta.tokenizers[t](p)
  print(f'Loaded tokenizer {t} with {len(p)} tokens')
  print(f'Tokeniser class {tokenizer.__class__}')
  if output is None:
    output=input + '.spm'
  all_tokens = []
  with open(input, encoding = 'utf-8') as fs:
    for l in tqdm(fs, ncols=80, desc='Loading'):
      if len(l) > 0:
        if vocab_type == 'custom':
          tokens = tokenizer.tokenize(l.strip(), max_seq_length)
        else:
          tokens = tokenizer.tokenize(l)
      else:
        tokens = []
      all_tokens.extend(tokens)

  print(f'Loaded {len(all_tokens)} tokens from {input}')
  lines = 0
  with open(output, 'w', encoding = 'utf-8') as wfs:
    idx = 0
    while idx < len(all_tokens):
      wfs.write(' '.join(all_tokens[idx:idx+max_seq_length-2]) + '\n')
      idx += (max_seq_length - 2)
      lines += 1

  print(f'Saved {lines} lines to {output}')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required=True, help='The input data path')
parser.add_argument('-o', '--output', default=None, help='The output data path')
parser.add_argument('--vocab_path', default=None, help='The vocabulary file path')
parser.add_argument('--max_seq_length', type=int, default=512, help='Maxium sequence length of inputs')
parser.add_argument('--vocab_type', default='custom', help='The type of vocabulary')
args = parser.parse_args()
tokenize_data(args.input, args.output, args.max_seq_length, args.vocab_path, args.vocab_type)
