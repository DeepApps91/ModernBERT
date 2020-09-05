import collections
import argparse
import pickle 
from glob import glob
from tqdm import tqdm


class Vocabulary(object):
    """Vocabulary object to numericalize or denumericalize a token"""

    def __init__(self,
                 input_file,
                 special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[MASK]"]):
        """Construct a dictionary from a corpus"""
        vocab = set()

        for file in tqdm(glob(input_file)):
            #print(f"Loading dictionary from {file}")
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip()
                    for char in line:
                        if char.strip():    # To avoid empty char
                            vocab.add(char)

        # Sort alphabetically and add special tokens on top 
        vocab = special_tokens + sorted(vocab)
        self.word2index = dict(zip(vocab, range(len(vocab))))
        self.index2word = {v: k for k, v in self.word2index.items()}

    def to_word(self, index):
        """Convert an index to its corresponding word"""
        if index in self.index2word:
            return self.index2word[index]
        else:
            return self.index2word[0]   # UNK_token

    def to_index(self, word):
        """Convert a word to its corresponding index"""
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.word2index["[UNK]"] 

    def save_vocab(self, output_file):
        """Serialize the Vocabulary object"""
        with open(output_file, 'wb') as f:
            pickle.dump(self, f)

    def load_vocab(self, input_file) -> 'Vocabulary':
        """Deserialize a Vocabulary object"""
        with open(input_file, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, metavar='PATH')
    parser.add_argument('--output_file', required=True, metavar='PATH')
    args = parser.parse_args()

    vocab = Vocabulary(args.input_file)
    vocab.save_vocab(args.output_file)
