# /home/lee/miniconda3/envs/elmo_pytorch/bin/python 
# -*- encoding = utf-8 -*-
# Author: Jiangtong Li

from typing import Callable, Optional, Tuple, List
from overrides import overrides
from module import CNN_EMBEDDING, ElmoLSTM, Highway, \
                   ElmoLSTM2, LstmCellWithProjection, \
                   Elmo_LSTM_BPE

import numpy as np

import torch 
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

# The model use subword as input and output the lm result or word embedding in [batch_size, sentence_length] format
class Bilm3(nn.Module):
    def __init__(self, opt):
        super(Bilm3, self).__init__()
        self.opt = opt
        self.dropout = torch.nn.Dropout(p=self.opt.drop_out_p)
        self.emb_dim = self.opt.subword_embedding_size
        self.embedding = nn.Embedding(self.opt.n_subwords, \
                                      self.opt.subword_embedding_size, \
                                      padding_idx=0)
        self.cnn = CNN_EMBEDDING(self.opt.subword_fliter, 
                                 self.opt.subword_embedding_size, 
                                 self.opt.cnn_activation)
        self.highway = Highway(self.opt.n_subword_fliter, 
                               self.opt.n_highway)
        self.linear = torch.nn.Linear(self.opt.n_subword_fliter, self.opt.projection_dim)
        self.Bilstm = Elmo_LSTM_BPE(self.opt.projection_dim,
                                    self.opt.projection_dim, 
                                    self.opt.dim,
                                    self.opt.num_layers,
                                    True,
                                    self.opt.drop_out_p,
                                    self.opt.cell_clips,
                                    self.opt.proj_clip, 
                                    self.opt.use_skip_connections)
        self.linear_f = torch.nn.Linear(self.opt.projection_dim, self.opt.vocab_size)
        self.linear_b = torch.nn.Linear(self.opt.projection_dim, self.opt.vocab_size)

    def forward(self, 
                input: torch.Tensor,
                batch_lengths: List[int],
                mask: torch.Tensor,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        '''
        The size of input is [seq_len, batch_size, num_token]
        '''
        input_size = list(input.size())
        #print(input_size)
        input = input.contiguous().view(-1, input_size[-1])
        input_size.append(self.emb_dim)
        embeds = self.embedding(input)
        embeds = self.dropout(embeds)
        cnn_out = self.cnn(embeds)
        cnn_out = self.dropout(cnn_out)
        high_out = self.highway(cnn_out)
        high_out = torch.transpose(high_out.view((input_size[0], input_size[1], -1)),
                                     1, 0)
        high_out = self.linear(high_out)
        high_out = high_out*mask.unsqueeze(-1)
        high_out = self.dropout(high_out)
        stacked_sequence_outputs, _final_state_tuple = self.Bilstm(high_out, \
                                                                  batch_lengths)
        if self.opt.mode == 'train':
            indices = torch.LongTensor([self.opt.num_layers-1]).cuda()
            final_sequence_outputs = torch.squeeze(
                        stacked_sequence_outputs.index_select(0, indices))
            a = list(range(0, self.opt.projection_dim))
            b = list(range(self.opt.projection_dim, 2*self.opt.projection_dim))
            indices_f = torch.LongTensor(a).cuda()
            indices_b = torch.LongTensor(b).cuda()
            f_sequence_outputs = self.dropout(final_sequence_outputs.index_select(-1, indices_f))
            b_sequence_outputs = self.dropout(final_sequence_outputs.index_select(-1, indices_b))
            f_prediction = self.linear_f(f_sequence_outputs)
            b_prediction = self.linear_b(b_sequence_outputs)
            return f_prediction, b_prediction
        elif self.opt.mode == 'use':
            return stacked_sequence_outputs
