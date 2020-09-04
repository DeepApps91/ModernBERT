# ModernBERT

gen_dict.py    

```
python gen_dict.py \
    --input_file='data/Corpus/*/*.txt' \
    --output_file='from_corpus.vocab'
```



text_processing.py    
load a corpus text and a dictionary. Convert the text to ids

train.py    
covert text to ids
divide to train, valid and test set    
train model    