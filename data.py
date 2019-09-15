import numpy as np
import random
from collections import OrderedDict
import pickle
import gzip

import sentencepiece as spm

from typing import List

def split_data(raw_file, out_file, piece_length):
    with open(raw_file, 'r') as fin:
        with open(out_file, 'w') as fout:
            for line in fin:
                line = line.strip('\n').split(' ')
                tmp = list()
                for item in line:
                    tmp.append(item)
                    if len(tmp) == piece_length:
                        tline = ' '.join(tmp) + '\n'
                        fout.write(tline)
                        tmp = list()
                if len(tmp) != 0:
                    tline = ' '.join(tmp) + '\n'
                    fout.write(tline)
                    tmp = list()

# The following code is for the standred BPE
def mk_subword_map(subword_file, pkl_file):
    """
    Build the BI-MAP between subword and index with some special encode mark
    
    Arguments:
        subword_file {str} -- The subword vocab file
        pkl_file {str} -- The file to store the processed MAP
    """
    # build vocab
    vocab = set()
    with open(subword_file) as f:
        for line in f:
            word = line.strip('\n').split('\t')[0]
            vocab.add(word)
    vocab = list(vocab)
    # build idx2subword
    idx2subword = list()
    idx2subword.append('__pad__')
    idx2subword.append('__unk__')
    idx2subword.append('__bos__')
    idx2subword.append('__eos__')
    idx2subword.append('__bow__')
    idx2subword.append('__eow__')
    for item in vocab:
        idx2subword.append(item)
    # bulid subword2idx
    subword2idx = dict()
    for idx, item in enumerate(idx2subword):
        subword2idx[item] = idx
    
    MAP = [subword2idx, idx2subword]
    with open(pkl_file, 'wb') as f:
        pickle.dump(MAP, f)

def mk_word_map(text_file, pkl_file_all, pkl_file_map, size):
    """
    Build the BI-MAP between word and index with some special encode mark
    
    Arguments:
        text_file {str} -- The raw text file
        pkl_file_map {str} -- The file to store the processed MAP(the first 5w word)
        pkl_file_all {str} -- The file to store the whole vocab with freq
    """
    # build the full vocab and order them in frequence
    word_freq = dict()
    with open(text_file) as f:
        for line in f:
            line = line.strip('\n').split(' ')
            for item in line:
                if item in word_freq:
                    word_freq[item] += 1
                else:
                    word_freq[item] = 1
    word_freq = OrderedDict(sorted(word_freq.items(), key=lambda t:t[1], reverse=True))
    with open(pkl_file_all, 'wb') as f:
        pickle.dump(word_freq, f)
    word_freq = list(word_freq.items())
    # build idx2word
    idx2word = list()
    idx2word.append('__pad__')
    idx2word.append('__unk__')
    idx2word.append('__bos__')
    idx2word.append('__eos__')
    for item in word_freq:
        if len(idx2word)>size:
            break
        idx2word.append(item[0])
    # build word2idx
    word2idx = dict()
    for idx, item in enumerate(idx2word):
        word2idx[item] = idx
    
    MAP = [word2idx, idx2word]
    with open(pkl_file_map, 'wb') as f:
        pickle.dump(MAP, f)

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode, encoding='utf-8')
    return open(filename, mode, encoding='utf-8')

class SubwordBPEBatcher(object):
    """
    The class is going to build the subword2idx map and 
    then transform a batch of sentences into a 4D numpy ndarray and 
    """
    def __init__(self, subword_pkl, word_pkl, raw_file=None,\
                 batch_size=32, word_length=10, \
                 shuffle=False, bpe_model=None):

        self.subword_pkl = subword_pkl
        self.word_pkl = word_pkl
        with open(self.subword_pkl, 'rb') as f:
            self.subword2idx, self.idx2subword = pickle.load(f)
        with open(self.word_pkl, 'rb') as f:
            self.word2idx, self.idx2word = pickle.load(f)

        self.raw_file = raw_file
        self.raw = fopen(self.raw_file)
        self.bpe_sp = spm.SentencePieceProcessor()
        self.bpe_sp.Load(bpe_model)

        self.word_length = word_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.end_of_data = False
        self.buffer = []
        self.k = self.batch_size*100
        
        self.left_raw = []
        self.left_length = []
        self.left_applied = []

    def encode_applied(self, applied):
        # transform the applied text into 3D matrix
        batch_size = len(applied)
        pad = ['__pad__'] * self.word_length
        applied_batch = np.zeros((batch_size, 20, self.word_length))
        applied_batch_mask = np.zeros((batch_size, 20))
        for idx1, item1 in enumerate(applied):
            for idx2, item2 in enumerate(item1):
                if len(item2)!=self.word_length:
                    item2 = item2[:self.word_length]
                item_idx = [self.subword2idx[i] for i in item2]
                applied_batch[idx1][idx2] = item_idx
                if item2 != pad:
                    applied_batch_mask[idx1][idx2] = 1
        return applied_batch, applied_batch_mask
    
    def encode_raw(self, raw):
        # transform the raw text into 2D matrix and give the length of each sentence
        batch_size = len(raw)
        raw_batch = np.zeros((batch_size, 20))
        for idx, item in enumerate(raw):
            item_idx = [self.word2idx[i] for i in item]
            raw_batch[idx] = item_idx
        return raw_batch

    def mkraw(self, raw):
        # add special mark and limit the length
        sentence = raw.strip('\n').split(' ')
        for idx, item in enumerate(sentence):
            if item not in self.word2idx:
                sentence[idx] = '__unk__'
        sentence.insert(0, '__bos__')
        sentence.append('__eos__')
        while len(sentence)%20 != 0:
            sentence.append('__pad__')
        return sentence

    def mkapplied(self, applied):
        # add special mark and limit the length
        pad = ['__pad__'] * self.word_length
        sentence = [('▁'+item).strip(' ').split(' ') \
                    for item in applied.strip('\n').split('▁') \
                    if item]
        sentence.insert(0, ['__bos__'])
        sentence.append(['__eos__'])
        for item in sentence:
            for idx, it in enumerate(item):
                if it not in self.subword2idx:
                    item[idx] = '__unk__'
            item.insert(0, '__bow__')
            item.append('__eow__')
            if len(item) > self.word_length:
                item = item[:self.word_length]
            while len(item)< self.word_length:
                item.append('__pad__')
            assert len(item) == self.word_length
        while len(sentence)%20 != 0:
            sentence.append(pad)
        length = len(sentence)
        return sentence, length

    def reset(self):
        self.raw.seek(0)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        if len(self.buffer) < self.batch_size:
            for _ in range(self.k):
                rr = self.raw.readline()
                aa = ' '.join(self.bpe_sp.EncodeAsPieces(rr))
                if rr == '':
                    break
                self.buffer.append((rr,aa))
            if self.shuffle:
                random.shuffle(self.buffer)
        if len(self.buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        raw = self.left_raw
        self.left_raw = []
        applied = self.left_applied
        self.left_applied = []
        length = self.left_length
        self.left_length = []

        try:
            while True:
                try:
                    sen_r, sen_a = self.buffer.pop(0)
                    if '�' in sen_r:
                        continue
                except IndexError:
                    break

                rr = self.mkraw(sen_r)
                aa, ll = self.mkapplied(sen_a)
                assert len(rr) == len(aa)
                while(ll >= 20):
                    raw.append(rr[:20])
                    applied.append(aa[:20])
                    length.append(20)
                    rr = rr[20:]
                    aa = aa[20:]
                    ll -= 20

                if len(raw) >= self.batch_size:
                    assert len(raw) == len(applied)
                    break
        except IOError:
            self.end_of_data = True

        if len(raw) > self.batch_size:
            self.left_raw = raw[self.batch_size:]
            raw = raw[:self.batch_size]
            self.left_applied = applied[self.batch_size:]
            applied = applied[:self.batch_size]
            self.left_length = length[self.batch_size:]
            length = length[:self.batch_size]

        if len(raw) < self.batch_size:
            assert len(raw) == len(applied)
            self.end_of_data = False
            self.reset()
            raise StopIteration

        assert len(raw) == len(applied)
        assert len(raw) == len(length)
        assert len(raw) == self.batch_size

        raw = self.encode_raw(raw)
        applied, mask = self.encode_applied(applied)
        return applied, length, raw, mask

class BPEBatcher(object):
    """
    The class is going to build the subword2idx map and 
    then transform a batch of sentences into a 4D numpy ndarray and 
    """
    def __init__(self, subword_pkl, word_pkl, raw_file, bpe_model, \
                 batch_size=32, word_length=10, sentence_length=50, \
                 shuffle=False):

        self.subword_pkl = subword_pkl
        self.word_pkl = word_pkl
        with open(self.subword_pkl, 'rb') as f:
            self.subword2idx, self.idx2subword = pickle.load(f)
        with open(self.word_pkl, 'rb') as f:
            self.word2idx, self.idx2word = pickle.load(f)

        self.raw_file = raw_file
        self.raw = fopen(self.raw_file)

        self.word_length = word_length
        self.sentence_length = sentence_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.bpe_sp = spm.SentencePieceProcessor()
        self.bpe_sp.Load(bpe_model)

        self.end_of_data = False
        self.raw_buffer = []
        self.applied_buffer = []
        self.k = self.batch_size*100

    def encode_applied(self, applied):
        # transform the applied text into 3D matrix
        batch_size = len(applied)
        pad = ['__pad__'] * self.word_length
        applied_batch = np.zeros((batch_size, 20, self.word_length))
        applied_batch_mask = np.zeros((batch_size, 20))
        for idx1, item1 in enumerate(applied):
            for idx2, item2 in enumerate(item1):
                if len(item2)!=self.word_length:
                    item2 = item2[:self.word_length]
                item_idx = [self.subword2idx[i] for i in item2]
                applied_batch[idx1][idx2] = item_idx
                if item2 != pad:
                    applied_batch_mask[idx1][idx2] = 1
        return applied_batch, applied_batch_mask
    
    def encode_raw(self, raw):
        # transform the raw text into 2D matrix and give the length of each sentence
        batch_size = len(raw)
        raw_batch = np.zeros((batch_size, 20))
        for idx, item in enumerate(raw):
            item_idx = [self.word2idx[i] for i in item]
            raw_batch[idx] = item_idx
        return raw_batch

    def mkraw(self, raw):
        # add special mark and limit the length
        sentence = raw.strip('\n').split(' ')
        for idx, item in enumerate(sentence):
            if item not in self.word2idx:
                sentence[idx] = '__unk__'
        sentence.insert(0, '__bos__')
        sentence.append('__eos__')
        while len(sentence)%20 != 0:
            sentence.append('__pad__')
        return sentence

    def mkapplied(self, applied):
        # add special mark and limit the length
        pad = ['__pad__'] * self.word_length
        sentence = [('▁'+item).strip(' ').split(' ') \
                    for item in applied.strip('\n').split('▁') \
                    if item]
        sentence.insert(0, ['__bos__'])
        sentence.append(['__eos__'])
        for item in sentence:
            for idx, it in enumerate(item):
                if it not in self.subword2idx:
                    item[idx] = '__unk__'
            item.insert(0, '__bow__')
            item.append('__eow__')
            if len(item) > self.word_length:
                item = item[:self.word_length]
            while len(item)< self.word_length:
                item.append('__pad__')
            assert len(item) == self.word_length
        while len(sentence)%20 != 0:
            sentence.append(pad)
        length = len(sentence)
        return sentence, length

    def reset(self):
        self.raw.seek(0)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        if len(self.raw_buffer) == 0:
            for _ in range(self.k):
                rr = self.raw.readline()
                aa = ' '.join(self.bpe_sp.EncodeAsPieces(rr))
                if rr == '':
                    break
                self.raw_buffer.append(rr)
                self.applied_buffer.append(aa)
            if self.shuffle:
                combine_list = list(zip(self.raw_buffer, self.applied_buffer))
                random.shuffle(combine_list)
                self.raw_buffer = [item[0] for item in combine_list]
                self.applied_buffer = [item[1] for item in combine_list]
            assert len(self.raw_buffer) == len(self.applied_buffer)
        if len(self.raw_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        raw = []
        applied = []
        length = []
        try:
            while True:
                try:
                    sen_r = self.raw_buffer.pop(0)
                    sen_a = self.applied_buffer.pop(0)
                except IndexError:
                    break

                rr = self.mkraw(sen_r)
                aa, ll = self.mkapplied(sen_a)
                assert len(rr) == len(aa), sen_a
                raw.append(rr)
                applied.append(aa)
                length.append(ll)

                if len(raw) >= self.batch_size:
                    assert len(raw) == len(applied), sen_r
                    break
        except IOError:
            self.end_of_data = True
            
        if len(raw) <= 0:
            assert len(raw) == len(applied), raw
            self.end_of_data = False
            self.reset()
            raise StopIteration
            
        assert len(raw) == len(applied)
        assert len(raw) == len(length)

        raw = self.encode_raw(raw)
        applied, mask = self.encode_applied(applied)
        return applied, length, raw, mask

def test():
    batcher = SubwordBPEBatcher(subword_pkl='toy_data/bpe_500_subword.pkl', word_pkl='toy_data/words_100000.pkl', \
                                raw_file='toy_data/raw', bpe_model='./toy_data/bpe_500.model')
    for idx in range(5):
        for i, some in enumerate(batcher):
            print("idx: %d, i:, %d" %(idx, i))


if __name__ == '__main__':
    test()
