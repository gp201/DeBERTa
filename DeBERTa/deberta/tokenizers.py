#
# Author: penhe@microsoft.com
# Date: 04/25/2019
#

""" tokenizers
"""

from .spm_tokenizer import *
from .gpt2_tokenizer import GPT2Tokenizer
from .custom_tokenizer import *

__all__ = ['tokenizers']
tokenizers={
    'gpt2': GPT2Tokenizer,
    'spm': SPMTokenizer,
    'custom': CustomTokenizer
    }
