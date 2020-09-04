import torch
from transformers import BertForMaskedLM, BertConfig


# load text and convert to ids

#define model
configuration = BertConfig(vocab_size = 7000)
model = BertForMaskedLM(configuration)
print(model.config)

#train model
