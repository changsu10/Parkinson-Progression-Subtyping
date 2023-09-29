import os
import time
import json
from itertools import product

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch import optim
from torch import nn
from torch.utils.data import *

import LSTM_AE

import paras

class PPMI_dataset(Dataset):
    def __init__(self, path, file_name, sequence_length):

        infile = open(path + file_name, 'rb')
        new_dict = pkl.load(infile)
        infile.close()

        temp_dict = {}
        for patient, mat in new_dict.items():
            if mat.shape[0] >= 5:
                temp_dict[patient] = mat
        new_dict = temp_dict

        count = 0
        for patient, mat in new_dict.items():
            count += 1
            if mat.shape[0] < sequence_length:
                num_row = sequence_length - mat.shape[0]
                add = np.zeros((num_row, mat.shape[1]))
                new_dict[patient] = np.append(mat, add, 0)

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


def training(path, file, model, hidden_size, num_layers, embed_size, batch_size, optimizer, LR, sequence_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training with %s..." % device)

    timestr = time.strftime("%Y%m%d")

    save_path = 'Output_' + model

    # load data-set
    dataset = PPMI_dataset(path, file, sequence_length)
    input_size = dataset.__featureNum__()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    # print(len(dataloader))
    # print(dataset.__len__())
    # print(dataset.__featureNum__())
    # term = dataset.__getitem__(4)
    # print(term[0], term[1][:, -5:])
    #
    # print('-----')
    #
    # infile = open(path + file, 'rb')
    # new_dict = pkl.load(infile)
    # infile.close()
    #
    # print(new_dict[3403][:, -5:])







    # define model
    model = eval(model).AutoEncoderRNN(input_size, hidden_size, num_layers, batch_size)
    model = model.float().to(device)

    # for param in model.parameters():
    #     param.requires_grad = True

    # define optimizer and generate save path
    if not os.path.isdir(f'{timestr}'):
        os.mkdir(f'{timestr}')

    if not os.path.isdir(f'{timestr}/{save_path}'):
        os.mkdir(f'{timestr}/{save_path}')

    if optimizer == 'SGD':
        Momentum = 0.9
        save_path = f'{timestr}/{save_path}/{file[:-4]}_hs{hidden_size}_emb{embed_size}_nly{num_layers}_opt{optimizer}_lr{LR}_m{Momentum}_b{batch_size}_seqL{sequence_length}/'

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=Momentum)


    if optimizer == 'Adam':
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        save_path = f'{timestr}/{save_path}/{file[:-4]}_hs{hidden_size}_emb{embed_size}_nly{num_layers}_opt{optimizer}_lr{LR}_1be{beta1}_2be{beta2}_eps{eps}_b{batch_size}_seqL{sequence_length}/'

        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        optimizer = optim.Adam(model.parameters(), lr=LR, betas=(beta1, beta2), eps=eps, amsgrad=False)

    # loss function
    criterion = nn.MSELoss()

    epochs = 10

    total_error = []
    for epoch in range(epochs+1):
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
                    vals = encoded[:, -1, :].reshape((batch_size, hidden_size))  # !!!! should be 0
                    for z in range(batch_size):
                        p = int(patient[z])
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
    path = '[your directory]/processed_data'
    file = 'sequence_data_LOCF&FOCB_imputation_Zscore.pkl'  # 'sequence_data_LOCF&FOCB_imputation_minmax.pkl', 'sequence_data_LOCF&FOCB_imputation_Zscore.pkl'

    model = 'LSTMAutov4_2'

    parameters = paras.hyperParas()
    parameters = parameters.__getParas__()
    with open('Output_' + model + '_' + 'parameters.json', 'w') as wf:
        json.dump(parameters, wf, indent=4)

    i = 0
    for par in [dict(zip(parameters, v)) for v in product(*parameters.values())]:

        if i > 1:
            continue

        sequence_length = par['sequence_length']
        embed_size = par['embed_size']
        hidden_size = par['hidden_size']
        num_layers = par['num_layers']
        batch_size = par['batch_size']
        LR = par['LR']
        optimizer = par['optimizer']

        if embed_size!= None and embed_size < hidden_size:
            continue

        i += 1

        print(model, hidden_size, num_layers, embed_size, batch_size, optimizer, LR, sequence_length)

        training(path, file, model, hidden_size, num_layers, embed_size, batch_size, optimizer, LR, sequence_length)




if __name__ == "__main__":
    main()