from typing import Callable, Optional, Tuple, List
from module.lstm_cell_with_projection import LstmCellWithProjection

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import torch 
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class ElmoLSTM(nn.Module):
    '''
    Parameters
    ----------
    input_size : ''int'', required
        The dimension of the inputs to the LSTM.
    hidden_size : ''int'', required
        The dimension of the outputs of the LSTM.
    cell_size : ''int'', required.
        The dimension of the memory cell of the
        :class:`~allennlp.modules.lstm_cell_with_projection.LstmCellWithProjection`.
    num_layers : ''int'', required
        The number of bidirectional LSTMs to use.
    requires_grad: ''bool'', optional
        If True, compute gradient of ELMo parameters for fine tuning.
    recurrent_dropout_probability: ''float'', optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    state_projection_clip_value: ''float'', optional, (default = None)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value: ''float'', optional, (default = None)
        The magnitude with which to clip the memory cell.
    '''
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 vocab_size: int, 
                 opt,
                 requires_grad: bool = True):
        super(ElmoLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.requires_grad = requires_grad
        self.opt = opt
        self.forward_lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=self.opt.drop_out_p)
        self.backward_lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=self.opt.drop_out_p)
        self.forward_lstm_cell = nn.ModuleList([nn.LSTMCell(input_size, hidden_size) for i in range(self.num_layers)])
        self.backward_lstm_cell = nn.ModuleList([nn.LSTMCell(input_size, hidden_size) for i in range(self.num_layers)])
        self.forward_linear = nn.Linear(hidden_size, vocab_size)
        self.backward_linear = nn.Linear(hidden_size, vocab_size)

    def forward(self,
                input_f: torch.Tensor = None,
                input_b: torch.Tensor = None,
                hidden_f: torch.Tensor = None, 
                hidden_b: torch.Tensor = None, 
                mark: str = 'train'):
        '''
        The shape of input is [seq_len, batch_size, input_size]
        '''
        total_sequence_length, batch_size, input_size = input_f.size()
        total_sequence_length = total_sequence_length
        input_size = input_size
        
        if hidden_f is None:
            hf = Variable(input_f.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float())
            cf = Variable(input_f.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float())
        else:
            hf, cf = hidden_f
        if hidden_b is None:
            hb = Variable(input_b.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float())
            cb = Variable(input_b.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float())
        else:
            hb, cb = hidden_b
        if  mark=='train':
            forward_output, _ = self.forward_lstm(input_f, (hf, cf))
            backward_output, _ = self.backward_lstm(input_b, (hb, cb))
            pred_forward = self.forward_linear(forward_output)
            pred_backword = self.backward_linear(backward_output)
            return pred_forward, pred_backword
        elif mark=='forward':
            forward_output, forward_hidden = self.forward_lstm(input_f, (hf,cf))
            pred_forward = self.forward_linear(forward_output)
            return forward_hidden, pred_forward
        elif mark=='backward':
            backward_output, backward_hidden = self.backward_lstm(input_b, (hb,cb))
            pred_backword = self.backward_linear(backward_output)
            return backward_hidden, pred_backword
        elif mark=='predict':
            forward_output, _ = self.forward_lstm(input_f, (hf, cf))
            backward_output, _ = self.backward_lstm(input_b, (hb, cb))
            return input_f, forward_output, backward_output
        else:
            raise ValueError('The mark must belongs to [\'represent\', \'train\', \'forward\', \'backward\']')

class ElmoLSTM2(nn.Module):
    '''
    Parameters
    ----------
    input_size : ''int'', required
        The dimension of the inputs to the LSTM.
    hidden_size : ''int'', required
        The dimension of the outputs of the LSTM.
    cell_size : ''int'', required.
        The dimension of the memory cell of the
        :class:`~allennlp.modules.lstm_cell_with_projection.LstmCellWithProjection`.
    num_layers : ''int'', required
        The number of bidirectional LSTMs to use.
    requires_grad: ''bool'', optional
        If True, compute gradient of ELMo parameters for fine tuning.
    recurrent_dropout_probability: ''float'', optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    state_projection_clip_value: ''float'', optional, (default = None)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value: ''float'', optional, (default = None)
        The magnitude with which to clip the memory cell.
    '''
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 vocab_size: int, 
                 opt,
                 requires_grad: bool = True):
        super(ElmoLSTM2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.requires_grad = requires_grad
        self.opt = opt
        self.forward_lstm_cell = nn.ModuleList([nn.LSTMCell(input_size, hidden_size) for i in range(self.num_layers)])
        self.backward_lstm_cell = nn.ModuleList([nn.LSTMCell(input_size, hidden_size) for i in range(self.num_layers)])
        self.forward_linear = nn.Linear(hidden_size, vocab_size)
        self.backward_linear = nn.Linear(hidden_size, vocab_size)
    def forward(self,
                input: torch.Tensor = None,
                hidden_f: torch.Tensor = None, 
                hidden_b: torch.Tensor = None, 
                mark: str = 'train'):
        '''
        The shape of input is [seq_len, batch_size, input_size]
        '''
        total_sequence_length, batch_size, input_size = input.size()
        total_sequence_length = total_sequence_length
        input_size = input_size
        hf = []
        cf = []
        hb = []
        cb = []
        
        if hidden_f is None:
            for idx in range(self.num_layers):
                hf.append(Variable(input.data.new(batch_size, self.hidden_size).fill_(0).float()))
                cf.append(Variable(input.data.new(batch_size, self.hidden_size).fill_(0).float()))
        else:
            hf, cf = hidden_f
        if hidden_b is None:
            for idx in range(self.num_layers):
                hb.append(Variable(input.data.new(batch_size, self.hidden_size).fill_(0).float()))
                cb.append(Variable(input.data.new(batch_size, self.hidden_size).fill_(0).float()))
        else:
            hb, cb = hidden_b
        if mark=='train':
            forward_hidden = []
            backward_hidden = []
            for i in range(total_sequence_length):
                input_ = input[i]
                for j in range(self.num_layers):
                    hf[j], cf[j] = self.forward_lstm_cell[j](input_, (hf[j], cf[j]))
                    input_ = hf[j]
                forward_hidden.append(input_)
            for i in range(total_sequence_length-1,-1,-1):
                input_ = input[i]
                for j in range(self.num_layers):
                    hb[j], cb[j] = self.backward_lstm_cell[j](input_, (hb[j], cb[j]))
                    input_ = hb[j]
                backward_hidden.insert(0, input_)
            forward_hidden = torch.stack(forward_hidden)
            backward_hidden = torch.stack(backward_hidden)
            forward_dis = self.forward_linear(forward_hidden)
            backward_dis = self.backward_linear(backward_hidden)
            return forward_dis, backward_dis
        elif mark=='forward':
            pass
        elif mark=='backward':
            pass
        elif mark=='represent':
            forward_hidden = []
            backward_hidden = []
            for i in range(total_sequence_length):
                each_for = []
                input_ = input[i]
                each_for.append(input_)
                for j in range(self.num_layers):
                    hf[j], cf[j] = self.forward_lstm_cell[j](input_, (hf[j], cf[j]))
                    each_for.append(hf[j])
                    input_ = hf[j]
                forward_hidden.append(torch.stack(each_for))
            for i in range(total_sequence_length-1,-1,-1):
                each_back = []
                input_ = input[i]
                each_back.append(input_)
                for j in range(self.num_layers):
                    hb[j], cb[j] = self.backward_lstm_cell[j](input_, (hb[j], cb[j]))
                    each_back.append(hb[j])
                    input_ = hf[j]
                backward_hidden.insert(0, torch.stack(each_back))
            forward_hidden = torch.stack(forward_hidden, 1)
            backward_hidden = torch.stack(backward_hidden, 1)
            return forward_hidden, backward_hidden
        else:
            raise ValueError('The mark must belongs to [\'represent\', \'train\', \'forward\', \'backward\']')