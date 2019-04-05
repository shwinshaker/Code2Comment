from copy import deepcopy
from collections import Counter
from functools import reduce
import ast
from ast2json import str2json
import pickle
import numpy as np


REMOVED = ['col_offset', 'lineno']
USER_DEFINED = ['name', 'id', 'attr', 'arg', 'module', 's', 'n']

class Traverse():
    def __init__(self):
        self.args = {}
        self.names = {}

    def __save_args(self, s):
        s_ = str(s)
        if s_ not in self.args:
            self.args[s_] = 1
        else:
            self.args[s_] += 1

    def __save_names(self, s):
        s_ = str(s)
        if s_ not in self.names:
            self.names[s_] = 1
        else:
            self.names[s_] += 1

    def __call__(self, node):
        if type(node) is list:
            for node_ in node:
                self(node_)
        elif type(node) is dict:
            for key in node:
                if key not in REMOVED:
                    self.__save_args(key)
                    if type(node[key]) not in [dict, list]:
                        if key in USER_DEFINED:
                            self.__save_names(node[key])
                        else:
                            self.__save_args(node[key])
                    else:
                        self(node[key])


class Sbt():
    def __init__(self, vocab):
        self.seq = []
        self.vocab = vocab

    def __remove_key(seld, d, key):
        d_ = deepcopy(d)
        del d_[key]
        return d_

    def __call__(self, node):
        if type(node) is list:
            if node:
                # appose
                # self.seq.append('(')
                for node_ in node:
                    self(node_)
            # self.seq.append(')')
        elif type(node) is dict:
            if node:
                key = list(node.keys())[0]
                # print(key)
                # print(node)
                if key.startswith('_'):
                    # self.seq.extend(['(', str(node[key])])
                    d_rest = self.__remove_key(node, key)
                    if not d_rest:
                        # the rest dict is null, only append key
                        self.seq.append(str(node[key]))
                    else:
                        # traverse the rest dict
                        self.seq.extend([str(node[key]), '('])
                        self(d_rest)
                        self.seq.append(')')
                else:
                    for key in node:
                        if not key in REMOVED:
                            # these two keys are not very useful
                            if type(node[key]) not in [dict, list]:
                                if key in USER_DEFINED:
                                    # these are self-defined names
                                    if str(node[key]) in self.vocab:
                                        # if name is common, include it
                                        self.seq.append('%s_%s' % (key, str(node[key])))
                                    else:
                                        # else if name is rare, only include the key
                                        self.seq.append(str(key))
                                else:
                                    self.seq.append('%s_%s' % (key, str(node[key])))
                            else:
                                if not node[key]:
                                    # the next node is null, either null dict or null list
                                    self.seq.append(str(key))
                                else:
                                    # traverse next node
                                    self.seq.extend([str(key), '('])
                                    # self.seq.extend(['(', str(key)])
                                    self(node[key])
                                    self.seq.append(')')


def build_vocab(filename='data/train.pkl', name_vocab_length=2000):
    with open(filename, 'rb') as f:
        text = pickle.load(f)

    def __print_common(common):
        print('\t', end='')
        for name, count in common:
            print('%s: %i' % (name, count), end=' ')
        print()

    def ___counter_length(c):
        _, values = zip(*c.most_common())
        return reduce(lambda x,y: x+y, values)

    print('----- frequently self-defined names ------')
    # build frequently-used self-defined vocab
    # limit to [id, name, attr, arg, s, n]
    traverse = Traverse()
    for i, (_, code) in enumerate(text):
        print(i, end='\r')
        traverse(str2json(code))
    c = Counter(traverse.names)
    assert(len(c.most_common()) == len(traverse.names))
    print('# names: %i' % ___counter_length(c))
    print('# unique names: %i' % len(c.most_common()))

    c_arg = Counter(traverse.args)
    print('# arguments: %i' % ___counter_length(c_arg))
    print('# unique arguments: %i' % len(c_arg.most_common()))

    # get the most frequently words as vocab
    c_vocab = Counter(dict(c.most_common(name_vocab_length)))
    print('Most frequent 5 names:')
    __print_common(c_vocab.most_common(5))
    print('Least frequent 5 names:')
    __print_common(c_vocab.most_common()[-5:])

    print('\nSelect the most frequent %i names' % name_vocab_length)

    # save for frequent use
    name_vocab, _ = zip(*c_vocab.most_common())
    with open('dicts/frequently_used_name.pkl', 'wb+') as f:
        pickle.dump(name_vocab, f)

    # why do we have to do two traverse?
    # the sbt treat frequently and rarely used names differently
    # maybe first do a traverse without discrimination
    # then find the word outside the vocab and remove thing after _?

    print('\n-------- SBT vocab ----------')
    sbt = Sbt(name_vocab)
    for i, (_, code) in enumerate(text):
        print(i, end='\r')
        sbt(str2json(code))
    c = Counter(sbt.seq)
    # add all traversed keys into vocabulary
    c.update(traverse.args)

    print('# sequences: %i' % i)
    print('# segments: %i' % len(sbt.seq))
    print('# unique segments: %i' % len(c.most_common()))
    print('Most frequent 5 segments:')
    __print_common(c.most_common(5))
    print('Least frequent 5 segments:')
    __print_common(c.most_common()[-5:])

    # save for frequent use
    segs, _ = zip(*c.most_common())
    vocab_dict = dict(zip(segs, range(len(segs))))
    with open('dicts/code_dict.pkl', 'wb+') as f:
        pickle.dump(vocab_dict, f)


class Encoder():
    def __init__(self):
        with open('dicts/code_dict.pkl', 'rb') as f:
            self.vocab_dict = pickle.load(f)
        with open('dicts/frequently_used_name.pkl', 'rb') as f:
            self.name_vocab = pickle.load(f)
        # append <EOS> to the dictionary
        assert('<EOS>' not in self.vocab_dict)
        assert(max(self.vocab_dict.values()) < len(self.vocab_dict))
        self.vocab_dict['<EOS>'] = len(self.vocab_dict)
        self.invert_dict = self.__invert(self.vocab_dict)

    def __invert(self, d):
        return dict([(y,x) for x, y in d.items()])

    def encode(self, code):
        sbt = Sbt(self.name_vocab)
        sbt(str2json(code))
        seq = []
        for token in sbt.seq:
            if token not in self.vocab_dict:
                if '_' in token:
                    token_ = token.split('_')[0]
                    if token_ not in self.vocab_dict:
                        raise KeyError('Some structural base not included! Critical!')
                    seq.append(self.vocab_dict[token_])
                    continue
                else:
                    raise KeyError('Some structural base not included! Critical!')
            seq.append(self.vocab_dict[token])
        seq.append(self.vocab_dict['<EOS>'])
        return seq

    def decode(self, seq):
        # to sbt tree
        return ''.join([self.invert_dict[index] for index in seq])


if __name__ == '__main__':
    # traverse the entire train set, build vocabulary for segments in sbt sequences
    # already built
    print('--- build vocab..')
    build_vocab()

    # check valid set
    # with open('data/valid.pkl', 'rb') as f:
    #     valid = pickle.load(f)
    # encoder = Encoder()
    # for i, (_, code) in enumerate(valid):
    #     print('valid [%i]' % i, end='\r')
    #     encoder.decode(encoder.encode(code))
    # print()

    # check test set
    # with open('data/test.pkl', 'rb') as f:
    #     test = pickle.load(f)
    # encoder = Encoder()
    # for i, (_, code) in enumerate(test):
    #     print('test [%i]' % i, end='\r')
    #     encoder.decode(encoder.encode(code))
    print()

    # encoding example
    # print('\n> Encode example:\n')
    with open('data/train.pkl', 'rb') as f:
        data = pickle.load(f)
    # Select one datus as exmaple
    comment, code = data[7755]
    encoder = Encoder()
    print('-------- Original code --------')
    print(code, end='\n\n')

    print('-------- sbt representation --------')
    print(encoder.decode(encoder.encode(code)), end='\n\n')

    print('-------- encoded code --------')
    print(encoder.encode(code))






