from typing import Callable, Optional, Tuple, List
from overrides import overrides

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Highway(nn.Module):
    """
    Parameters
    ----------
    input_dim : ''int''
        The dimensionality of :math:`x`.  We assume the input has shape ''(batch_size,
        input_dim)''.
    num_layers : ''int'', optional (default=''1'')
        The number of highway layers to apply to the input.
    activation : ''Callable[[torch.Tensor], torch.Tensor]'', optional (default=''torch.nn.functional.relu'')
        The non-linearity to use in the highway layers.
    """
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1,
                 activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu):
        super(Highway, self).__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim * 2)
                                            for _ in range(num_layers)])
        self._activation = activation
        for layer in self._layers:
            layer.bias[input_dim:].data.fill_(1)

    @overrides
    def forward(self, inputs: torch.Tensor):
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part = projected_input[:, (0 * self._input_dim):(1 * self._input_dim)]
            gate = projected_input[:, (1 * self._input_dim):(2 * self._input_dim)]
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input
