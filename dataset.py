# This file is based on the implementation of huggingface's LineByLineTextDataset 
# It is changed to:
#       - Read multiple input files with different encodings
#       - Disable adding special tokens
#       - Enable padding

import os
import pickle
import time
from typing import Optional
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers import PreTrainedTokenizer
from transformers import logging


logger = logging.get_logger(__name__)


class JapaneseDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, encoding:str="utf-8"):
        files = glob(file_path)
        assert len(files) > 0, f"No input file match pattern {file_path}"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        lines = []
        for file in tqdm(files, desc="Loading input files..."):
            with open(file, encoding=encoding) as f:
                lines.extend(line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()))

        batch_encoding = tokenizer(lines, add_special_tokens=False, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)
