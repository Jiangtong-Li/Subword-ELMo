from data import BPEBatcher

import torch
from torch.autograd import Variable

from progressbar import *
import numpy as np
import h5py

total = 132751

def raw2embedding(raw_file, subword_file, word_file, bpe_model_file, elmo_model, output_file, batch_size):
    data = BPEBatcher(subword_file, word_file, raw_file, bpe_model_file, batch_size=batch_size, word_length=6, sentence_length=None, shuffle=False)
    Model = torch.load(elmo_model, map_location=lambda storage, loc: storage)
    Model.opt.mode = 'use'
    Model.cuda()
    Model.eval()
    h5f = h5py.File(output_file, 'w')
    idx = 0
    pbar = ProgressBar().start()
    for applied, length, raw, mask in data:
        input = Variable(torch.Tensor(applied).long().transpose(1,0)).contiguous().cuda()
        mask = Variable(torch.Tensor(mask).float()).contiguous().cuda()
        embedding = Model(input, length, mask)
        embedding = torch.transpose(embedding, 1, 0)[0].cpu()
        embedding_np = embedding.detach().numpy()[-1, 1:-1, :]
        h5f.create_dataset(str(idx), data=embedding_np)
        idx += 1
        pbar.update(int((idx / (total - 1)) * 100))
    pbar.close()
    h5f.close()

if __name__ == '__main__':
    raw_file = '/home/lijt/Lab/pytorch_bpe/data/train_raw'
    subword_file = '/home/lijt/Lab/pytorch_bpe/data/vocab_1000_all/subwords_1000.pkl'
    word_file = '/home/lijt/Lab/pytorch_bpe/data/words.pkl'
    bpe_model_file = '/home/lijt/Lab/pytorch_bpe/data/vocab_1000_all/model_bpe_1000_all.model'
    elmo_model = '/home/lijt/Lab/pytorch_bpe/model/epoch_13_1000.pth'
    output_file =  '/home/lijt/Lab/pytorch_bpe/out/conll2009_epoch_13_1000.h5'
