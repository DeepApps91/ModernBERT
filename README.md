# ModernBERT

Sentence scoring with BERT.

This BERT implementation has the following modifications:

- Input contains only a single sentence.
- Discard the NSP task and only train on MLM objective.
- [MASK] token is replaced 100% of the time instead of 80% in the original BERT.
- Remove special tokens [CLS] and [SEP].



## Create vocab.txt file from corpus

```bash
python gen_dict.py \
    --input_file="data/*.txt" \
    --output_file="vocab.txt"
```

**Remark:** passing the --help flag to see all possible arguments.



## Train BERT from scratch

```bash
python run_language_modeling.py \
    --max_position_embeddings=64 \
    --train_data_file="data/*.txt" \
    --output_dir="output/" \
    --do_train \
    --save_steps=100 \
    --max_steps=300 \
    --seed=12345
```

**Remark:** passing the --help flag to see all possible arguments.