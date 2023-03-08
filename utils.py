import torch
import numpy as np
import random
import torch.utils.data as utils
import pandas as pd


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)

bnp_data = pd.read_pickle("frame_of_speed_matrix.pickle")

def create_BNP(inputs_R):
    Tindex = inputs_R[:,:,0:1]
    output0 = torch.empty(1,inputs_R.shape[1],410,410)
    for i in range(inputs_R.shape[0]-1):
        output1 = torch.empty(1,1,410,410)
        for j in range(inputs_R.shape[1]-1):
            output1 = torch.cat((output1,torch.tensor(bnp_data.iloc[int(Tindex[i,j,0]),0][0]).unsqueeze(0).unsqueeze(0)), dim=1)
        output0 = torch.cat((output0, output1), dim=0)
    return output0


def PrepareDataset(main_passenger_matrix, BATCH_SIZE=20, pred_len=1, train_propotion=0.7, valid_propotion=0.2):
    time_len = main_passenger_matrix.shape[0]

    max_load = main_passenger_matrix.max().max()
    main_passenger_matrix = main_passenger_matrix / max_load
    main_passenger_matrix.insert(0,'index',range(1,len(main_passenger_matrix)+1),False)

    # Weekly-periodic data preparation
    pass_sequences, pass_labels = [], []

    # Number of historical sequence
    seq_len = 2

    for i in range(time_len - 109 * 7 * (seq_len + pred_len)):
        pass_sequences.append(main_passenger_matrix.iloc[[i + 109 * 7 * j for j in range(seq_len)]].values)  #109 is the number of steps between two consecutive days
        pass_labels.append(main_passenger_matrix.iloc[[i + 109 * 7 * (seq_len + j) for j in range(pred_len)]].values)

    pass_sequences, pass_labels = np.asarray(pass_sequences), np.asarray(pass_labels)

    sample_size = pass_sequences.shape[0]

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    # randomize the order of sequences

    c = list(zip(pass_sequences, pass_labels))
    random.shuffle(c)
    pass_sequences, pass_labels = zip(*c)

    train_data, train_label = pass_sequences[:train_index], pass_labels[:train_index]
    valid_data, valid_label = pass_sequences[train_index:valid_index], pass_labels[train_index:valid_index]
    test_data, test_label = pass_sequences[valid_index:], pass_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset_w = utils.TensorDataset(train_data, train_label)
    valid_dataset_w = utils.TensorDataset(valid_data, valid_label)
    test_dataset_w = utils.TensorDataset(test_data, test_label)
    # Daily-periodic data preparation

    seq_len = 4
    pass_sequences, pass_labels = [], []

    # Specify the time range from which to get the data
    maind = main_passenger_matrix.loc['2021-09-23 05:00:00':'2021-10-04 23:00:00']

    time_len = maind.shape[0]

    for i in range(time_len - 109 * (seq_len + pred_len)):
        pass_sequences.append(maind.iloc[[i + 109 * j for j in range(seq_len)]].values)
        pass_labels.append(maind.iloc[[i + 109 * (seq_len + j) for j in range(pred_len)]].values)

    pass_sequences, pass_labels = np.asarray(pass_sequences), np.asarray(pass_labels)

    sample_size = pass_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    # randomize the order of sequences

    c = list(zip(pass_sequences, pass_labels))
    random.shuffle(c)
    pass_sequences, pass_labels = zip(*c)

    train_data, train_label = pass_sequences[:train_index], pass_labels[:train_index]
    valid_data, valid_label = pass_sequences[train_index:valid_index], pass_labels[train_index:valid_index]
    test_data, test_label = pass_sequences[valid_index:], pass_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset_d = utils.TensorDataset(train_data, train_label)
    valid_dataset_d = utils.TensorDataset(valid_data, valid_label)
    test_dataset_d = utils.TensorDataset(test_data, test_label)

    # Recent data preparation

    seq_len = 10
    pass_sequences, pass_labels = [], []

    # Specify the time range from which to get the data
    mainr = main_passenger_matrix.loc['2021-09-26 21:30:00':'2021-10-04 05:00:00']

    time_len = mainr.shape[0]
    for i in range(time_len - (seq_len + pred_len)):
        pass_sequences.append(mainr.iloc[[i + j for j in range(seq_len)]].values)
        pass_labels.append(mainr.iloc[[i + (seq_len + j) for j in range(pred_len)]].values)




    pass_sequences, pass_labels = np.asarray(pass_sequences), np.asarray(pass_labels)

    sample_size = pass_sequences.shape[0]
    index = np.arange(sample_size, dtype=int)
    np.random.shuffle(index)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    # randomize the order of sequences

    c = list(zip(pass_sequences, pass_labels))
    random.shuffle(c)
    pass_sequences, pass_labels = zip(*c)

    train_data, train_label = pass_sequences[:train_index], pass_labels[:train_index]
    valid_data, valid_label = pass_sequences[train_index:valid_index], pass_labels[train_index:valid_index]
    test_data, test_label = pass_sequences[valid_index:], pass_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset_r = utils.TensorDataset(train_data, train_label)
    valid_dataset_r = utils.TensorDataset(valid_data, valid_label)
    test_dataset_r = utils.TensorDataset(test_data, test_label)

    # All the data in one dataloader

    train_loader = torch.utils.data.DataLoader(
        ConcatDataset(train_dataset_r, train_dataset_d, train_dataset_w),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(
        ConcatDataset(valid_dataset_r, valid_dataset_d, valid_dataset_w),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        ConcatDataset(test_dataset_r, test_dataset_d, test_dataset_w),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_loader, valid_loader, test_loader, max_load
