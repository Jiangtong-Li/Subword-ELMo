from typing import Callable, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.autograd import Variable

class CNN_EMBEDDING(nn.Module):
    """
    Parameters
    ----------
    fliter : ''list''
        The size of the fliter to use in the cnn layers
    activation : ''Callable[[torch.Tensor], torch.Tensor]'', optional (default=''torch.nn.functional.relu'')
        The non-linearity to use in the cnn layers.
    """
    def __init__(self,
                 fliter: list, 
                 input_dim: int, 
                 activation = torch.nn.functional.relu):
        super(CNN_EMBEDDING, self).__init__()
        # if self.opt.cnn_activation == 'relu':
        #     activation = nn.ReLU(inplace=True)
        # elif self.opt.cnn_activation == 'tanh':
        #     activation = nn.Tanh()
        self.activation = activation(inplace=True)
        self.cnn_group = torch.nn.ModuleList([nn.Conv1d(in_channels=input_dim, out_channels=num, kernel_size=width, stride=1, bias=True)
                                            for (width, num) in fliter])

    def forward(self, input:torch.Tensor):
        input = torch.transpose(input, 1, 2)
        cnn_out = []
        for cnn in self.cnn_group:
            cnn_tmp = cnn(input)
            cnn_tmp, _ = torch.max(cnn_tmp, -1)
            cnn_out.append(self.activation(cnn_tmp))
            #print(cnn_tmp.size())
        out = torch.cat(cnn_out, -1)
        return out
