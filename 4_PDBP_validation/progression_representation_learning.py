"""
_*_ coding: utf-8 _*_
@ Author: Yu Hou
@File: progression_representation_learning.py
@Time: 1/13/21 9:59 PM
"""

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import time


class Prepare_dataset:
    def __init__(self, path, file_name, sequence_length):

        infile = open(path + file_name, 'rb')
        new_dict = pkl.load(infile)
        infile.close()

        temp_dict = {}
        for patient, mat in new_dict.items():
            if mat.shape[0] >= 3:  # >= 1year  PDBP: >= 3
                temp_dict[patient] = mat
        new_dict = temp_dict

        count = 0
        for patient, mat in new_dict.items():
            count += 1

            if mat.shape[0] >= sequence_length:
                newmat = mat[:sequence_length, :]
                new_dict[patient] = newmat
        self.samples = new_dict

    def __len__(self):
        return len(self.samples)

    def __featureNum__(self):
        return self.__getitem__(0)[1].shape[1]

    def __getitem__(self, idx):
        return [list(self.samples)[idx], list(self.samples.values())[idx]]


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
                Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(
                    device))  # <- change here: first dim of hidden needs to be doubled

    def forward(self, x, batch_size):

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

    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
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


class Training:

    def __init__(self, path, file, hidden_size, num_layers, embed_size, batch_size, optimizer, LR, sequence_length, output_path):
        self.path = path
        self.file = file
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.LR = LR
        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.optimizer = optimizer
        self.output_path = output_path

    def training(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training with %s..." % device)

        timestr = time.strftime("%Y%m%d")

        save_path = self.output_path

        # load data-set
        dataset = Prepare_dataset(self.path, self.file, self.sequence_length)
        input_size = dataset.__featureNum__()

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                                 drop_last=True)

        # define model
        model = AutoEncoderRNN(input_size, self.hidden_size, self.num_layers, self.batch_size)
        model = model.float().to(device)

        # define optimizer and generate save path
        # if not os.path.isdir(f'{timestr}'):
        #     os.mkdir(f'{timestr}')

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        if self.optimizer == 'SGD':
            Momentum = 0.9
            save_path = f'{save_path}/{self.file[:-4]}_hs{self.hidden_size}_emb{self.embed_size}_nly{self.num_layers}_opt{self.optimizer}_lr{self.LR}_m{Momentum}_b{self.batch_size}_seqL{self.sequence_length}/'

            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            optimizer = optim.SGD(model.parameters(), lr=self.LR, momentum=Momentum)

        if self.optimizer == 'Adam':
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            save_path = f'{save_path}/{self.file[:-4]}_hs{self.hidden_size}_emb{self.embed_size}_nly{self.num_layers}_opt{self.optimizer}_lr{self.LR}_1be{beta1}_2be{beta2}_eps{eps}_b{self.batch_size}_seqL{self.sequence_length}/'

            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            optimizer = optim.Adam(model.parameters(), lr=self.LR, betas=(beta1, beta2), eps=eps, amsgrad=False)

        # loss function
        criterion = nn.MSELoss()

        epochs = 50

        total_error = []
        for epoch in range(epochs + 1):
            error = []
            for i, data in enumerate(dataloader):
                model = model.train()

                patient, inputs = data[0], data[1].float().to(device)

                enc, pred = model(inputs)

                optimizer.zero_grad()
                loss = criterion(inputs.float(), pred)
                # print(loss)
                loss.backward()
                optimizer.step()
                error.append(loss.data.cpu().numpy())

            # print(loss.data.cpu().numpy())

            print("Epoch: %s, loss: %s..." % (epoch, np.mean(error)))
            if epoch % 10 == 0:
                model.eval()
                patient_encode = {}
                patient_full_hidden = {}
                with torch.no_grad():
                    for j, data in enumerate(dataloader):
                        patient, inputs = data[0], data[1].float().to(device)
                        enc, _ = model(inputs)

                        encoded = enc.data.cpu().numpy()
                        vals = encoded[:, -1, :].reshape((self.batch_size, self.hidden_size))  # !!!! should be 0
                        for z in range(self.batch_size):
                            p = patient[z]
                            # p = int(patient[z])
                            v = vals[z, :]
                            full_v = encoded[z, :, :]
                            patient_encode[p] = v
                            patient_full_hidden[p] = full_v

                    # with open(save_path + '/' + 'embedding_' + 'epoch_%s' % epoch + '.pkl', 'wb') as wf:
                    #     pkl.dump(patient_encode, wf)
                    with open(save_path + '/' + 'full_hidden_' + 'epoch_%s' % epoch + '.pkl', 'wb') as wf:
                        pkl.dump(patient_full_hidden, wf)

            total_error.append(np.mean(error))
        plt.plot(total_error)
        plt.savefig(save_path + '/' + 'loss.pdf')
        plt.close()


def main():
    path = '[your directory]/validation/processed_data/'
    file = 'sequence_data_Zscore.pkl'
    output_path = '[your directory]/validation/LSTM_Output/'

    hidden_size = 32
    num_layers = 1
    embed_size = None
    batch_size = 1
    optimizer = 'Adam'
    LR = 0.001
    sequence_length = 10
    Train = Training(path, file, hidden_size, num_layers, embed_size, batch_size, optimizer, LR, sequence_length, output_path)
    Train.training()


if __name__ == '__main__':
    main()
