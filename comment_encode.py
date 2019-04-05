import string
from collections import Counter
import re
import pickle

import string
from collections import Counter
import re
import pickle


def comment2list(comment, regex=re.compile('[%s0-9]' % re.escape(string.punctuation))):
    return [w.lower() for w in regex.sub('', comment).split()]


def build_vocab(file_name='data/train.pkl', vocab_length=3000):
    # load data
    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    # comments only
    comments, _ = zip(*data)

    # traverse
    word_list = []
    for comment in comments:
        word_list.extend(comment2list(comment))
    c = Counter(word_list)
    most_common_words, most_common_words_count = zip(*c.most_common(vocab_length))
    print('5 most common words:', most_common_words[:5])
    print('5 most common words count', most_common_words_count[:5])
    comment_dict = dict(zip(most_common_words, range(len(most_common_words))))

    # add additional annotations
    for token in ['<EOS>', '<SOS>', '<UNK>']:
        assert(token not in comment_dict)
        assert(max(comment_dict.values()) < len(comment_dict))
        comment_dict[token] = len(comment_dict)

    with open('dicts/comment_dict.pkl', 'wb') as pfile:
        pickle.dump(comment_dict, pfile)


class Encoder():
    def __init__(self):
        with open('dicts/comment_dict.pkl', 'rb') as f:
            self.vocab_dict = pickle.load(f)

        self.invert_dict = self.__invert(self.vocab_dict)

    def __invert(self, d):
        return dict([(y,x) for x, y in d.items()])

    def encode(self, comment): 
        comment_list = ['<SOS>']
        comment_list.extend(comment2list(comment))
        comment_list.append('<EOS>')
        comment_list = [w if w in self.vocab_dict else '<UNK>' for w in comment_list]
        return [self.vocab_dict[w] for w in comment_list]

    def decode(self, seq):
        # to sentence
        return ' '.join([self.invert_dict[index] for index in seq])


if __name__ == '__main__':

    print('--- build vocab..')
    build_vocab()
    print()

    with open('data/train.pkl', 'rb') as f:
        data = pickle.load(f)
    # Select one datus as exmaple
    comment, _ = data[7755]
    encoder = Encoder()
    print('-------- Original comment --------')
    print(comment, end='\n\n')

    print('-------- encoded comment --------')
    print(encoder.encode(comment), end='\n\n')

    print('-------- decoded comment --------')
    print(encoder.decode(encoder.encode(comment)))


