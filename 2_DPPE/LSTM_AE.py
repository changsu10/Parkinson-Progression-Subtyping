import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
	
	def __init__(self, input_size, hidden_size, num_layers, bidirectional):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
							dropout=0, bidirectional=bidirectional)
		self.sigmoid = nn.Sigmoid()

		nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

	def init_hidden(self, batch_size):
		return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device),
				Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device))  # <- change here: first dim of hidden needs to be doubled

	def forward(self, x, batch_size):
		# set initial hidden and cell states
		# h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
		# c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

		self.hidden = self.init_hidden(batch_size)

		# forward propagate lstm
		out, (hn, cn) = self.lstm(x, self.hidden)  # out: tensor of shape (batch_size, seq_length, hidden_size)
		out = torch.flip(out, [1])

		newinput = torch.flip(x, [1])

		zeros = torch.zeros(batch_size, 1, x.shape[-1])

		newinput = torch.cat((zeros, newinput), 1)

		newinput = newinput[:, :-1, :]

		return out, (hn, cn), newinput

class DecoderRNN(nn.Module):

	def __init__(self,  input_size, hidden_size, num_layers, bidirectional):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.input_size =input_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
							dropout=0, bidirectional=bidirectional)

		nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
		nn.init.orthogonal_(self.lstm.weight_hh_l0, gain=np.sqrt(2))

		
	def forward(self, x, h, batch_size):
		# forward propagate lstm
		out, _ = self.lstm(x, h)  # out: tensor of shape (batch_size, seq_length, hidden_size)

		fin = torch.flip(out, [1])

		return fin

class AutoEncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, batch_size, bidirectional=False):
		super(AutoEncoderRNN, self).__init__()

		self.batch_size = batch_size

		self.encoder = EncoderRNN(input_size, hidden_size, num_layers, bidirectional)
		self.decoder = DecoderRNN(input_size, hidden_size, num_layers, bidirectional)

		self.linear = nn.Linear(hidden_size, input_size)
		nn.init.orthogonal_(self.linear.weight, gain=np.sqrt(2))

	def forward(self, x):
		encoded_x, (hn, cn), newinput = self.encoder(x, self.batch_size)
		decoded_x = self.decoder(newinput, (hn, cn), self.batch_size)
		decoded_x = self.linear(decoded_x)
		# print(torch.sum(self.linear.weight.data))
		return encoded_x, decoded_x

