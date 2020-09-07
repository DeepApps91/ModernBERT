# ModernBERT

Use `gen_dict.py` to generate a dictionary from a corpus. The output file is a serialized Vocabulary object.

```bash
python gen_dict.py \
    --input_file='data/Corpus/*/*.txt' \
    --output_file='vocab'
```

Use pickle to deserialize 

```python
from gen_dict import Vocabulary
import pickle

with open("vocab", "rb") as f:
	vocab = pickle.load(f)
```



text_processing.py    
load a corpus text and a dictionary. Convert the text to ids

train.py    
covert text to ids
divide to train, valid and test set    
train model    
