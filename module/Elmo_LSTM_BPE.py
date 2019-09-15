from typing import Callable, Optional, Tuple, List
from module.lstm_cell_with_projection import LstmCellWithProjection

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import torch 
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class Elmo_LSTM_BPE(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 cell_size: int,
                 num_layers: int,
                 requires_grad: bool = False,
                 recurrent_dropout_probability: float = 0.0,
                 memory_cell_clip_value: Optional[float] = None,
                 state_projection_clip_value: Optional[float] = None,
                 use_skip_connection: bool = True):
        super(Elmo_LSTM_BPE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_size = cell_size
        self.requires_grad = requires_grad
        self.use_skip_connection = use_skip_connection

        forward_layers = []
        backward_layers = []

        lstm_input_size = input_size
        go_forward = True

        for layer_index in range(num_layers):
            forward_layer = LstmCellWithProjection(lstm_input_size,
                                                   hidden_size,
                                                   cell_size,
                                                   go_forward,
                                                   recurrent_dropout_probability,
                                                   memory_cell_clip_value,
                                                   state_projection_clip_value)
            backward_layer = LstmCellWithProjection(lstm_input_size,
                                                    hidden_size,
                                                    cell_size,
                                                    not go_forward,
                                                    recurrent_dropout_probability,
                                                    memory_cell_clip_value,
                                                    state_projection_clip_value)
            lstm_input_size = hidden_size

            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)
        self.forward_layers = forward_layers
        self.backward_layers = backward_layers

    def forward(self, 
                inputs: Variable, 
                length: List[int], 
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        if initial_state is None:
            hidden_states = [None] * len(self.forward_layers)
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), \
                                     initial_state[1].split(1, 0)))
        
        forward_output_sequence = inputs
        backward_output_sequence = inputs

        final_states = []
        sequence_outputs = []

        for layer_index, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))
            backward_layer = getattr(self, 'backward_layer_{}'.format(layer_index))

            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence

            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
                forward_state = (forward_hidden_state, forward_memory_state)
                backward_state = (backward_hidden_state, backward_memory_state)
            else:
                forward_state = None
                backward_state = None

            forward_output_sequence, forward_state = forward_layer(forward_output_sequence,
                                                                   length, 
                                                                   forward_state)
            backward_output_sequence, backward_state = backward_layer(backward_output_sequence, 
                                                                      length,
                                                                      backward_state)

            # Skip connections, just adding the input to the output.
            if layer_index != 0 and self.use_skip_connection:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache
            
            sequence_outputs.append(torch.cat([forward_output_sequence,
                                               backward_output_sequence], -1))
            
            # Append the state tuples in a list, so that we can return
            # the final states for all the layers.
            final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
                                 torch.cat([forward_state[1], backward_state[1]], -1)))

        stacked_sequence_outputs = torch.stack(sequence_outputs)
        # Stack the hidden state and memory for each layer into 2 tensors of shape
        # (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
        # respectively.
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple = (torch.cat(final_hidden_states, 0),
                                                       torch.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple

