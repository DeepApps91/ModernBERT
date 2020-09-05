import os
import glob
import collections
import argparse


def load_vocab(filename):
    """Load dictionary from existing vocabulary files"""
    word2index = collections.OrderedDict()

    with open(filename, 'r') as f:
        print(f"Loading dictionary from {filename}")
        for line in f:
            line = line.strip()
            word, index = line.split()
            word2index[word] = int(index)

    print(f"Done loading dictionary from {filename}")
    
    return word2index


def load_corpus(filename, start_index):
    """Load dictionary from a corpus"""
    vocab = set()

    for file in glob.glob(filename):
        print(f"Loading dictionary from {file}")
        with open(file, 'r') as f:
            for line in f:
                line = line.strip()
                for char in line:
                    if char.strip():    # To avoid empty char
                        vocab.add(char)

    # Create an alphabetically sorted dictionary of vocabulary 
    vocab_dict = zip(sorted(vocab), range(start_index, len(vocab)+start_index))
    word2index = collections.OrderedDict(vocab_dict)

    print(f"Done loading dictionary from corpus")

    return word2index 


class Vocabulary:
    """Create a dictionary from a corpus or an existing vocabulary file"""

    def __init__(self, input_file, from_vocab=False):
        self.word2index = None

        if from_vocab:
            self.word2index = load_vocab(input_file)
        else:
            # We ignore the [CLS] and [SEP] tokens of the original BERT.
            # According to the paper, they are not helpful for the masked  
            # token prediction task.
            self.word2index = collections.OrderedDict({"[UNK]": 0,
                                                       "[SOS]": 1,
                                                       "[EOS]": 2,
                                                       "[PAD]": 3,
                                                       "[MASK]": 4})
            self.word2index.update(load_corpus(input_file,
                                               len(self.word2index)))

        self.index2word = {v: k for k, v in self.word2index.items()}
        self.num_word = len(self.word2index)

    def add_word(self, word):
        """Add new word to dictionary"""

        if word not in self.word2index:
            self.word2index[word] = self.num_word
            self.index2word[self.num_word] = word
            self.num_word += 1

    def to_word(self, index):
        """Convert a word to an index"""

        if index in self.index2word:
            return self.index2word[index]
        else:
            return self.index2word[0]   # UNK_token

    def to_index(self, word):
        """Convert an index to a word"""

        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.word2index["[UNK]"] 

    def to_file(self, output_file):
        """Export dictionary to a vocabulary file"""

        with open(output_file, 'w') as f:
            for index in range(self.num_word):
                f.write(f"{self.index2word[index]}\t{index}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dictionary from a corpus or an existing vocabulary file")

    parser.add_argument('--input_file',
                        required=True,
                        metavar='PATH',
                        help='path to the input file(s) (allow multiple input files if load from a corpus and only one input file if load from a vocab file)')

    parser.add_argument('--output_file',
                        required=True,
                        metavar='PATH',
                        help='path to the output vocab file')

    parser.add_argument('--from_vocab',
                        action='store_true',
                        default=False,
                        help='create dictionary from vocab instead of corpus (default: False)')

    args = parser.parse_args()

    vocab = Vocabulary(args.input_file, args.from_vocab)
    vocab.to_file(args.output_file)
