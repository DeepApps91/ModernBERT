import collections
import argparse
import pickle 
from glob import glob
from tqdm import tqdm


class Vocabulary(object):
    """Vocabulary object to numericalize or denumericalize a token.
    Attributes:
        index2word: A list of words sorted in alphabetical order
        word2index: A dictionary maps from word -> index
        size: Size of the vocabulary
    """

    def __init__(self,
                 input_file,
                 encoding='utf-8',
                 special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[MASK]"]):
        """Construct a dictionary from a corpus"""
        vocab = set()

        for file in tqdm(glob(input_file)):
            try:
                with open(file, 'r', encoding=encoding) as f:
                    for line in f:
                        line = line.strip()
                        for char in line:
                            if char.strip():    # To avoid empty char
                                vocab.add(char)
            except UnicodeDecodeError as e:
                print(e)
                print("UnicodeDecodeError at file:", file)

        # Sort alphabetically and add special tokens on top 
        vocab = special_tokens + sorted(vocab)
        self.index2word = vocab
        self.word2index = dict(zip(vocab, range(len(vocab))))
        self.size = len(self.word2index)

    def to_words(self, indexes):
        """Convert list of indexes to theirs corresponding words"""
        words = []
        for index in indexes:
            print(index)
            if index >= 0 and index < self.size:
                words.append(self.index2word[index])
            else:
                words.append(self.index2word[0])   # UNK_token

        return words

    def to_indexes(self, words):
        """Convert list of words to theirs corresponding indexes"""
        indexes = []
        for word in words:
            if word in self.word2index:
                indexes.append(self.word2index[word])
            else:
                indexes.append(self.word2index["[UNK]"])

        return indexes

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
    parser.add_argument('--encoding', default='utf-8', metavar='ENCODING')

    args = parser.parse_args()

    vocab = Vocabulary(args.input_file, args.encoding)
    vocab.save_vocab(args.output_file)

    print(f"Vocabulary size={vocab.size}")
